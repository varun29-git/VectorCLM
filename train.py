import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from collections import deque
from datasets import load_dataset
from tokenizers import Tokenizer

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset
import random
import math
from muon import SingleDeviceMuon
import bitsandbytes as bnb

TOTAL_TRAINING_TOKENS = 3_500_000_000
WARMUP_STEPS = 1000

MODEL_FOLDER = "checkpoints"


def get_adamw_lr(step, total_steps):
    """Cosine LR schedule with linear warmup for the AdamW optimizer."""
    if step < WARMUP_STEPS:
        # Linear warmup: 0 → LR over WARMUP_STEPS
        return LR * (step / WARMUP_STEPS)
    # Cosine decay after warmup
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = max(0.0, min(1.0, progress))
    min_lr = 1e-5
    return min_lr + 0.5 * (LR - min_lr) * (1 + math.cos(math.pi * progress))


def get_model(vocab_size):
    return build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )


def split_parameters(model):
    """
    Split model parameters into two groups:
      - Muon group: All 2D weight matrices inside decoder blocks
        (attention projections and MLP weights). Skip embeddings and LM head.
      - AdamW group: All 1D parameters (biases, RMSNorm gains),
        the Embedding table, and the LM Head.
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D weights inside decoder layers → Muon
        if param.ndim >= 2 and "decoder.layers" in name:
            muon_params.append(param)
        else:
            # 1D params (RMSNorm gamma), embedding, lm_head → AdamW
            adamw_params.append(param)

    return muon_params, adamw_params


def train_mixed_strategy(model, optimizer_muon, optimizer_adamw, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load tokenizer.json ({e})")
        return

    # Total steps for LR schedule
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    total_steps = TOTAL_TRAINING_TOKENS // tokens_per_step

    # Load Datasets
    print("Loading datasets with streaming...")

    def keep_text_only(ds):
        return ds.select_columns(["text"])

    # Cosmopedia
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
    ds_cosmo = keep_text_only(ds_cosmo)

    # FineWeb-Edu
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds_fineweb = keep_text_only(ds_fineweb)

    # Weights
    probabilities = [0.6, 0.4]

    print("\n" + "=" * 50)
    print("DATASET CONFIGURATION")
    print("=" * 50)
    print(f"1. HuggingFaceTB/cosmopedia (Web)       : {probabilities[0]*100}%")
    print(f"2. HuggingFaceFW/fineweb-edu (Edu)      : {probabilities[1]*100}%")
    print("=" * 50 + "\n")

    from datasets import interleave_datasets

    print(f"Interleaving datasets with probabilities: {probabilities}")
    mixed_dataset = interleave_datasets(
        [ds_cosmo, ds_fineweb],
        probabilities=probabilities,
        seed=42,
        stopping_strategy="first_exhausted"
    )

    # DataLoader
    dl = DataLoader(
        StreamingLanguageModelDataset(mixed_dataset, SEQ_LEN, tokenizer),
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=True
    )
    iterator = iter(dl)

    pbar = tqdm(total=TOTAL_TRAINING_TOKENS // (BATCH_SIZE * SEQ_LEN), dynamic_ncols=True)
    loss_window = deque(maxlen=50)

    optimizer_muon.zero_grad(set_to_none=True)
    optimizer_adamw.zero_grad(set_to_none=True)

    step = 0
    opt_step = 0  # Counts actual optimizer steps (after grad accum)

    model.train()

    while global_tracker['tokens_seen'] < TOTAL_TRAINING_TOKENS:
        step += 1

        try:
            batch = next(iterator)
        except StopIteration:
            print("Dataset exhausted. Restarting iterator...")
            iterator = iter(dl)
            batch = next(iterator)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_tokens = input_ids.numel()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            opt_step += 1

            # Gradient clipping (over all parameters)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update AdamW LR via cosine schedule with warmup
            current_adamw_lr = get_adamw_lr(opt_step, total_steps)
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = current_adamw_lr

            # Muon LR stays constant at 0.02 (set at init)

            # Step both optimizers
            optimizer_muon.step()
            optimizer_adamw.step()

            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)

        # Updates
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(1)

        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)

        current_adamw_lr = get_adamw_lr(opt_step, total_steps)
        pbar.set_postfix({
            "AdamW_LR": f"{current_adamw_lr:.1e}",
            "Muon_LR": "2.0e-02",
            "L": f"{avg_loss:.2f}",
        })

    pbar.close()
    print(f"Training Complete. Total Tokens: {global_tracker['tokens_seen']:,}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For Mac MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    vocab_size = VOCAB_SIZE
    model = get_model(vocab_size).to(device)

    # --- Parameter Grouping ---
    muon_params, adamw_params = split_parameters(model)

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    n_total = n_muon + n_adamw
    print(f"Model Parameters: {n_total:,}")
    print(f"  Muon group  (2D decoder weights): {n_muon:,} ({len(muon_params)} tensors)")
    print(f"  AdamW group (1D + embed + head) : {n_adamw:,} ({len(adamw_params)} tensors)")

    # --- Dual Optimizer Initialization ---
    optimizer_muon = SingleDeviceMuon(
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True
    )

    optimizer_adamw = bnb.optim.AdamW8bit(
        adamw_params,
        lr=LR,  # 6e-4
        betas=(0.90, 0.95),
        weight_decay=WEIGHT_DECAY  # 0.1
    )

    # Global Progress Tracker
    global_tracker = {
        'start_time': time.time(),
        'tokens_seen': 0
    }

    train_mixed_strategy(
        model=model,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        vocab_size=vocab_size,
        global_tracker=global_tracker
    )

    torch.save(model.state_dict(), f"{MODEL_FOLDER}/model_final.pt")

    total_time = time.time() - global_tracker['start_time']
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Total Time: {total_time/3600:.2f} hours")
    print(f"Total Tokens Processed: {global_tracker['tokens_seen']:,}")
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
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

TOTAL_TRAINING_TOKENS = 3_000_000_000
WARMUP_STEPS = 1000

MODEL_FOLDER = "checkpoints"


def get_adamw_lr(step, total_steps):
    """Cosine LR schedule with linear warmup for the AdamW optimizer."""
    if step < WARMUP_STEPS:
        return LR * (step / WARMUP_STEPS)
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
        if param.ndim >= 2 and "decoder.layers" in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def get_curriculum_probs(tokens_seen):
    """Returns dataset probabilities based on 4-phase curriculum (3B limit)."""
    # Datasets: [TinyStories, Cosmo-Stories, Cosmo-v2, FineWeb-Edu, ProofWriter, TinyCodes]
    if tokens_seen < 50_000_000:
        # Phase 1 (0 to 50M tokens): Basic Language & Simple Logic
        return [0.45, 0.40, 0.00, 0.00, 0.15, 0.00], "Phase 1: Basic"
    elif tokens_seen < 1_500_000_000:
        # Phase 2 (50M to 1.5B tokens): Knowledge & Reasoning
        return [0.015, 0.015, 0.55, 0.25, 0.10, 0.07], "Phase 2: Knowledge"
    elif tokens_seen < 2_500_000_000:
        # Phase 3 (1.5B to 2.5B tokens): Applied Logic & Code
        return [0.00, 0.00, 0.25, 0.40, 0.15, 0.20], "Phase 3: Logic/Code"
    else:
        # Phase 4 (2.5B to 3.0B tokens): Cooldown & Consolidation
        return [0.00, 0.00, 0.10, 0.30, 0.30, 0.30], "Phase 4: Cooldown"


def safe_get_batch(iterator, dataloader, dataset_name):
    """Robust batch fetcher with shard exhaustion fallback."""
    try:
        return next(iterator), iterator
    except StopIteration:
        print(f"Dataset {dataset_name} exhausted. Restarting stream...")
        iterator = iter(dataloader)
        try:
            return next(iterator), iterator
        except StopIteration:
            raise RuntimeError(f"CRITICAL: Dataset {dataset_name} is empty or broken.")


def train_mixed_strategy(model, optimizer_muon, optimizer_adamw, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load tokenizer.json ({e})")
        return

    # Total steps for AdamW LR schedule
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    total_steps = TOTAL_TRAINING_TOKENS // tokens_per_step

    # --- Dataset Loading and Standardization ---
    print("Loading curriculum datasets with streaming...")

    # 1. TinyStories
    ds_tiny = load_dataset("roneneldan/TinyStories", split="train", streaming=True).select_columns(["text"])
    
    # 2. Cosmopedia Stories (General Version)
    ds_stories = load_dataset("HuggingFaceTB/cosmopedia", name="stories", split="train", streaming=True).select_columns(["text"])
    
    # 3. Cosmopedia-v2
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True).select_columns(["text"])
    
    # 4. FineWeb-Edu
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).select_columns(["text"])
    
    # 5. ProofWriter
    ds_proof = load_dataset("tasksource/proofwriter", split="train", streaming=True)
    ds_proof = ds_proof.map(lambda x: {"text": f"Context:\n{x['theory']}\n\nQuestion: {x['question']}\nAnswer: {str(x['answer'])}"}).select_columns(["text"])
    
    # 6. TinyCodes (nampdn-ai/tiny-codes)
    ds_codes = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
    ds_codes = ds_codes.filter(lambda x: x["programming_language"] == "Python")
    ds_codes = ds_codes.map(lambda x: {"text": f"Question:\n{x['prompt']}\n\nCode:\n{x['response']}"}).select_columns(["text"])

    datasets = [ds_tiny, ds_stories, ds_cosmo, ds_fineweb, ds_proof, ds_codes]
    dataset_names = ["TinyStories", "Cosmo-Stories", "Cosmo-v2", "FineWeb-Edu", "ProofWriter", "TinyCodes"]

    # --- DataLoaders and Iterators ---
    dataloaders = [
        DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), 
                   batch_size=BATCH_SIZE, num_workers=4, prefetch_factor=2, pin_memory=True)
        for ds in datasets
    ]
    iterators = [iter(dl) for dl in dataloaders]

    pbar = tqdm(total=TOTAL_TRAINING_TOKENS // (BATCH_SIZE * SEQ_LEN), dynamic_ncols=True)
    loss_window = deque(maxlen=50)

    # Watchdog Variables
    best_loss = float('inf')
    last_improvement_time = time.time()
    lr_multiplier = 1.0

    optimizer_muon.zero_grad(set_to_none=True)
    optimizer_adamw.zero_grad(set_to_none=True)

    step = 0
    opt_step = 0
    model.train()

    while global_tracker['tokens_seen'] < TOTAL_TRAINING_TOKENS:
        step += 1

        # Curriculum Routing
        current_probs, phase_name = get_curriculum_probs(global_tracker['tokens_seen'])
        dataset_idx = random.choices(range(len(datasets)), weights=current_probs, k=1)[0]
        
        # Safe Batch Fetching
        batch, iterators[dataset_idx] = safe_get_batch(iterators[dataset_idx], dataloaders[dataset_idx], dataset_names[dataset_idx])

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_tokens = input_ids.numel()

        # Inlined forward + backward
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            opt_step += 1
            
            # --- Emergency Defibrillator Watchdog ---
            avg_loss = sum(loss_window) / len(loss_window) if loss_window else loss.item()
            if avg_loss > 3.0:
                if avg_loss < best_loss - 0.2:
                    best_loss = avg_loss
                    last_improvement_time = time.time()
                
                # Check for 2-hour stagnation (7200 seconds)
                if time.time() - last_improvement_time > 7200:
                    print("\nSTALL DETECTED: Applying 1.5x Thermal Kick")
                    lr_multiplier *= 1.5
                    best_loss = avg_loss # Reset baseline
                    last_improvement_time = time.time() # Reset timer
            
            # Update AdamW LR with multiplier
            current_adamw_lr = get_adamw_lr(opt_step, total_steps) * lr_multiplier
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = current_adamw_lr

            # Update Muon LR with multiplier
            for param_group in optimizer_muon.param_groups:
                param_group['lr'] = 0.02 * lr_multiplier

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_muon.step()
            optimizer_adamw.step()

            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)

        # Updates
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(1)

        loss_window.append(loss.item())
        current_avg_loss = sum(loss_window) / len(loss_window)

        pbar.set_postfix({
            "Phase": phase_name,
            "Adam_LR": f"{current_adamw_lr:.1e}" if opt_step > 0 else "0.0e+00",
            "L": f"{current_avg_loss:.2f}",
            "Watchdog": f"{lr_multiplier:.1f}x" if current_avg_loss > 3.0 else "OFF"
        })

    pbar.close()
    print(f"Training Complete. Total Tokens: {global_tracker['tokens_seen']:,}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    vocab_size = VOCAB_SIZE
    model = get_model(vocab_size).to(device)

    # Compiled model only
    print("Compiling model (only forward pass)...")
    model = torch.compile(model, mode="reduce-overhead")

    # --- Parameter Grouping ---
    muon_params, adamw_params = split_parameters(model)
    
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"Model Parameters: {n_muon + n_adamw:,}")

    # --- Separate Dual Optimizers ---
    optimizer_muon = SingleDeviceMuon(muon_params, lr=0.02, momentum=0.95, nesterov=True)
    optimizer_adamw = bnb.optim.AdamW8bit(adamw_params, lr=6e-4, betas=(0.90, 0.95), weight_decay=0.1)

    # Global Progress Tracker
    global_tracker = {'start_time': time.time(), 'tokens_seen': 0}

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
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
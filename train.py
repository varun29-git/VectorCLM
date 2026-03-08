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

TOTAL_TRAINING_TOKENS = 50_000_000_000

def get_lr(tokens_seen):
    # Simple Cosine Decay for 100M tokens
    progress = tokens_seen / TOTAL_TRAINING_TOKENS
    progress = max(0.0, min(1.0, progress))
    
    target_min = 1e-5
    target_max = LR
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return target_min + (target_max - target_min) * cosine_decay

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

# Define mapping functions used in dataset loading

def train_mixed_strategy(model, optimizer, scaler, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Define mapping functions
    def map_tiny_codes(x):
        prompt = x.get('prompt', '')
        response = x.get('response', '')
        text = f"User: {prompt}\n\nAssistant: {response}"
        return {"text": text}

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load tokenizer.json ({e})")
        return

    # Dataset Configs
    # Total: 2B


    # Load Datasets
    print("Loading datasets with streaming...")
    
    # helper to ensure we only have 'text' column to avoid interleaving schema conflicts
    def keep_text_only(ds):
        return ds.select_columns(["text"])

    # Cosmopedia 
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
    ds_cosmo = keep_text_only(ds_cosmo)
    
    # FineWeb-Edu 
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds_fineweb = keep_text_only(ds_fineweb)
    
    # Tiny Codes 
    ds_code = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
    ds_code = ds_code.map(map_tiny_codes) 
    ds_code = keep_text_only(ds_code)

    # DCLM 
    ds_dclm = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
    ds_dclm = keep_text_only(ds_dclm)

    # Weights
    probabilities = [0.5, 0.3, 0.1, 0.1]
    
    print("\n" + "="*50)
    print("DATASET CONFIGURATION")
    print("="*50)
    print(f"1. HuggingFaceTB/cosmopedia (Web)       : {probabilities[0]*100}%")
    print(f"2. HuggingFaceFW/fineweb-edu (Edu)      : {probabilities[1]*100}%")
    print(f"3. nampdn-ai/tiny-codes (Code)          : {probabilities[2]*100}%")
    print(f"4. mlfoundations/dclm-baseline-1.0      : {probabilities[3]*100}%")
    print("="*50 + "\n")
    
    from datasets import interleave_datasets
    
    # Interleave
    print(f"Interleaving datasets with probabilities: {probabilities}")
    mixed_dataset = interleave_datasets(
        [ds_cosmo, ds_fineweb, ds_code, ds_dclm],
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
    optimizer.zero_grad(set_to_none=True)
    step = 0
    
    model.train()
    
    while global_tracker['tokens_seen'] < TOTAL_TRAINING_TOKENS:
        step += 1
        
        # LR Schedule (Global)
        current_lr = get_lr(global_tracker['tokens_seen'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        try:
            batch = next(iterator)
        except StopIteration:
            print("Dataset exhausted. Restarting iterator...")
            iterator = iter(dl)
            batch = next(iterator)
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_tokens = input_ids.numel()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward() # type: ignore

        if step % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Updates
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(1)

        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)
        
        pbar.set_postfix({
            "LR": f"{current_lr:.1e}",
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
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Initialize Optimizer
    optimizer = None
    try:
        import bitsandbytes as bnb
        print("Using 8-bit AdamW optimizer via bitsandbytes...")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.99),
            eps=1e-8
        )
    except Exception as e:
        print(f"Warning: bitsandbytes failed to load ({e}). Fallback to standard AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.99),
            eps=1e-8
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    # Global Progress Tracker
    global_tracker = {
        'start_time': time.time(),
        'tokens_seen': 0
    }

    train_mixed_strategy(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
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
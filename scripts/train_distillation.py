#!/usr/bin/env python3
"""
Knowledge Distillation Training.
Trains student model (33M) using teacher model responses.
Loss = α * CrossEntropy + β * KL Divergence
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import ChartLanguageModel


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    
    def __init__(self, teacher_data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load teacher responses
        with open(teacher_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} teacher-student pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        teacher_response = item["teacher_response"]
        
        # Format as: "Prompt: {prompt}\nResponse: {response}"
        full_text = f"Prompt: {prompt}\nResponse: {teacher_response}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": prompt,
            "teacher_response": teacher_response
        }


def distillation_loss(
    student_logits,
    student_labels,
    teacher_logits=None,
    temperature=3.0,
    alpha=0.5,
    beta=0.5
):
    """
    Compute distillation loss.
    Loss = α * CE(student, labels) + β * KL(student, teacher)
    """
    # Cross-entropy loss (student vs ground truth)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        student_labels.view(-1),
        ignore_index=-100
    )
    
    # KL divergence loss (if teacher logits available)
    kl_loss = torch.tensor(0.0, device=student_logits.device)
    if teacher_logits is not None:
        # Softmax with temperature
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (temperature ** 2)
    
    # Combined loss
    total_loss = alpha * ce_loss + beta * kl_loss
    
    return total_loss, ce_loss, kl_loss


def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1, temperature=3.0, alpha=0.5, beta=0.5):
    """Train for one epoch with distillation."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Distillation Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Create labels (shift by 1 for next token prediction)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token
        
        # Distillation loss
        # Note: We don't have teacher logits, so we use CE only (alpha=1.0, beta=0.0)
        # For full distillation, you'd need teacher model logits
        loss, ce_loss, kl_loss = distillation_loss(
            logits,
            labels,
            teacher_logits=None,  # Would need teacher model for this
            temperature=temperature,
            alpha=1.0,  # Use only CE since no teacher logits
            beta=0.0
        )
        
        # Scale for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_ce_loss += ce_loss.item() * accumulation_steps
        total_kl_loss += kl_loss.item() * accumulation_steps
        
        progress_bar.set_postfix({
            "loss": f"{loss.item() * accumulation_steps:.4f}",
            "ce": f"{ce_loss.item() * accumulation_steps:.4f}",
            "kl": f"{kl_loss.item() * accumulation_steps:.4f}"
        })
    
    return {
        "total_loss": total_loss / len(dataloader),
        "ce_loss": total_ce_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader)
    }


def train_distillation(config_path="./configs/distillation_config.yaml"):
    """Main distillation training function."""
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 70)
    print()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", "./tokenizer")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pretrained student model
    pretrained_path = model_config["pretrained_model_path"]
    print(f"Loading pretrained student model from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    model = ChartLanguageModel(
        vocab_size=vocab_size,
        d_model=model_config.get("d_model", 512),
        n_layers=model_config.get("n_layers", 8),
        n_heads=model_config.get("n_heads", 8),
        d_ff=model_config.get("d_ff", 2048),
        max_seq_len=model_config.get("max_seq_len", 256),
        dropout=model_config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    num_params = model.get_num_params()
    print(f"Student model loaded: {num_params:,} parameters ({num_params/1e6:.2f}M)")
    print()
    
    # Load distillation data
    teacher_data_path = data_config["teacher_data_path"]
    print(f"Loading teacher data from {teacher_data_path}...")
    
    if not os.path.exists(teacher_data_path):
        print(f"❌ Teacher data not found: {teacher_data_path}")
        print("   Run collect_teacher_data.py first!")
        return
    
    dataset = DistillationDataset(teacher_data_path, tokenizer, max_length=model_config.get("max_seq_len", 256))
    
    batch_size = training_config.get("batch_size", 2)
    num_workers = training_config.get("num_workers", 2)
    if device.type in ["cpu", "mps"]:
        num_workers = min(num_workers, 2)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Setup optimizer
    learning_rate = training_config.get("learning_rate", 1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    # Output directory
    output_dir = training_config.get("output_dir", "./models/distilled")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    num_epochs = training_config.get("num_epochs", 10)
    accumulation_steps = training_config.get("gradient_accumulation_steps", 16)
    temperature = training_config.get("temperature", 3.0)
    alpha = training_config.get("alpha", 0.5)
    beta = training_config.get("beta", 0.5)
    
    print(f"Starting distillation training for {num_epochs} epochs...")
    print(f"Temperature: {temperature}, Alpha: {alpha}, Beta: {beta}")
    print()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        losses = train_epoch(model, dataloader, optimizer, device, accumulation_steps, temperature, alpha, beta)
        
        print(f"  Total Loss: {losses['total_loss']:.4f}")
        print(f"  CE Loss: {losses['ce_loss']:.4f}")
        print(f"  KL Loss: {losses['kl_loss']:.4f}")
        print()
        
        # Save checkpoint
        if (epoch + 1) % training_config.get("save_every", 1) == 0:
            checkpoint_path = os.path.join(output_dir, f"distilled_epoch_{epoch + 1}.pt")
            try:
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Save checkpoint
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": losses['total_loss'],
                    "config": config
                }, checkpoint_path)
                
                # Verify checkpoint was saved
                if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
                    print(f"✅ Checkpoint saved: {checkpoint_path}")
                    print(f"   File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
                else:
                    print(f"❌ ERROR: Checkpoint file was not created or is empty: {checkpoint_path}")
                    print(f"   Attempting to save again...")
                    # Retry once
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": losses['total_loss'],
                        "config": config
                    }, checkpoint_path)
                    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
                        print(f"✅ Checkpoint saved on retry: {checkpoint_path}")
                    else:
                        print(f"❌ ERROR: Failed to save checkpoint after retry!")
            except Exception as e:
                print(f"❌ ERROR: Failed to save checkpoint for epoch {epoch + 1}: {str(e)}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            print()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size
    }, final_model_path)
    
    print("=" * 70)
    print(f"✅ Distillation training complete!")
    print(f"   Final model saved to: {final_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with knowledge distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/distillation_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train_distillation(args.config)


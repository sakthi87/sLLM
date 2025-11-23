"""
Fine-tune model on domain-specific data (logs, metadata).
Optimized for high-memory Linux systems (120GB RAM).
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import ChartLanguageModel
from utils.data_preparation import load_text_from_logs, load_text_from_metadata, collect_all_texts


class DomainDataset(Dataset):
    """Dataset for domain-specific fine-tuning."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def load_domain_data(config: dict):
    """Load domain-specific data (logs, metadata)."""
    all_texts = []
    
    # Load log data
    logs_dir = config.get("domain_logs_path", "./data/domain_logs")
    if os.path.exists(logs_dir):
        for log_file in Path(logs_dir).glob("*.json"):
            texts = load_text_from_logs(str(log_file))
            all_texts.extend(texts)
            print(f"Loaded {len(texts)} examples from {log_file.name}")
    
    # Load metadata
    metadata_dir = config.get("domain_metadata_path", "./data/domain_metadata")
    if os.path.exists(metadata_dir):
        for meta_file in Path(metadata_dir).glob("*.json"):
            texts = load_text_from_metadata(str(meta_file))
            all_texts.extend(texts)
            print(f"Loaded {len(texts)} examples from {meta_file.name}")
    
    return all_texts


def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader)


def finetune_domain(config_path: str = "./configs/domain_finetune_linux.yaml"):
    """Main domain fine-tuning function."""
    print("=" * 70)
    print("Domain-Specific Fine-Tuning")
    print("=" * 70)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", "./tokenizer")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Load base model
    base_model_path = model_config.get("base_model_path")
    if base_model_path and os.path.exists(base_model_path):
        print(f"Loading base model from {base_model_path}...")
        checkpoint = torch.load(base_model_path, map_location=device)
        
        model = ChartLanguageModel(
            vocab_size=vocab_size,
            d_model=model_config.get("d_model", 512),
            n_layers=model_config.get("n_layers", 8),
            n_heads=model_config.get("n_heads", 8),
            d_ff=model_config.get("d_ff", 2048),
            max_seq_len=model_config.get("max_seq_len", 512),
            dropout=model_config.get("dropout", 0.1),
            pad_token_id=tokenizer.pad_token_id
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded base model")
    else:
        # Initialize from scratch if no base model
        print("Initializing new model...")
        model = ChartLanguageModel(
            vocab_size=vocab_size,
            d_model=model_config.get("d_model", 512),
            n_layers=model_config.get("n_layers", 8),
            n_heads=model_config.get("n_heads", 8),
            d_ff=model_config.get("d_ff", 2048),
            max_seq_len=model_config.get("max_seq_len", 512),
            dropout=model_config.get("dropout", 0.1),
            pad_token_id=tokenizer.pad_token_id
        )
    
    model = model.to(device)
    print(f"Model has {model.get_num_params():,} parameters")
    
    # Load domain data
    print("\nLoading domain-specific data...")
    texts = load_domain_data(data_config)
    
    if len(texts) == 0:
        print("Error: No domain data found!")
        return
    
    print(f"Loaded {len(texts)} domain examples")
    
    # Create dataset
    dataset = DomainDataset(texts, tokenizer, max_length=model_config.get("max_seq_len", 512))
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 64),
        shuffle=True,
        num_workers=training_config.get("num_workers", 8),
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-5),
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    # Output directory
    output_dir = training_config.get("output_dir", "./models/domain_tuned")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    num_epochs = training_config.get("num_epochs", 10)
    accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device, accumulation_steps)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size
    }, final_path)
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Fine-tuning complete! Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune on domain data")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/domain_finetune_linux.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    finetune_domain(args.config)


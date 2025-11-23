"""
Instruction tuning: Fine-tune the pretrained model on conversational data.
This teaches the model to follow instructions and have conversations.
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
from utils.data_preparation import load_text_from_chat_data, collect_all_texts


class InstructionDataset(Dataset):
    """Dataset for instruction tuning on conversational data."""
    
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


def load_instruction_data(data_dir: str, focus_on_chat: bool = True):
    """Load data prioritizing conversational/instruction examples."""
    all_texts = []
    
    # Prioritize chat data
    chat_dir = os.path.join(data_dir, "chat")
    if os.path.exists(chat_dir):
        for file in Path(chat_dir).rglob("*.json"):
            texts = load_text_from_chat_data(str(file))
            all_texts.extend(texts)
            print(f"Loaded {len(texts)} instruction examples from {file}")
    
    # Also include other data but with less weight
    if not focus_on_chat:
        # Load logs and metadata too
        logs_dir = os.path.join(data_dir, "logs")
        if os.path.exists(logs_dir):
            for file in Path(logs_dir).rglob("*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        text = f"Log: {item.get('log_entry', '')}\n"
                        text += f"Question: {item.get('question', '')}\n"
                        text += f"Answer: {item.get('answer', '')}\n"
                        all_texts.append(text)
    
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


def instruction_tune(config_path: str = "./configs/instruction_tune_config.yaml"):
    """Main instruction tuning function."""
    print("=" * 50)
    print("Instruction Tuning - Conversational Fine-tuning")
    print("=" * 50)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    
    # Setup device (optimized for M1 Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Check if we should use CPU instead
    use_cpu = training_config.get("use_cpu", False)
    if use_cpu:
        device = torch.device("cpu")
        print("Forcing CPU mode (use_cpu=True in config)")
    
    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", "./tokenizer")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pretrained model
    pretrained_path = model_config["pretrained_model_path"]
    print(f"Loading pretrained model from {pretrained_path}...")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    model = ChartLanguageModel(
        vocab_size=vocab_size,
        d_model=model_config.get("d_model", 768),
        n_layers=model_config.get("n_layers", 12),
        n_heads=model_config.get("n_heads", 12),
        d_ff=model_config.get("d_ff", 3072),
        max_seq_len=model_config.get("max_seq_len", 512),
        dropout=model_config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Load pretrained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"Model loaded with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Load instruction data
    print("Loading instruction/conversational data...")
    conv_data_path = data_config.get("conversational_data_path")
    
    if conv_data_path and os.path.exists(conv_data_path):
        # Load from collected conversational file
        print(f"Loading from {conv_data_path}...")
        with open(conv_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = []
            for item in data:
                if "messages" in item:
                    # Format conversation
                    conv_text = ""
                    for msg in item["messages"]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        conv_text += f"{role.capitalize()}: {content}\n\n"
                    texts.append(conv_text.strip())
    else:
        # Fallback to old method
        texts = load_instruction_data(
            data_config.get("data_dir", "./data"),
            focus_on_chat=data_config.get("focus_on_chat", True)
        )
    
    if len(texts) == 0:
        print("Error: No instruction data found!")
        return
    
    print(f"Loaded {len(texts)} instruction examples")
    
    # Create dataset and dataloader
    dataset = InstructionDataset(texts, tokenizer, max_length=model_config.get("max_seq_len", 512))
    batch_size = training_config.get("batch_size", 16)
    num_workers = training_config.get("num_workers", 4)
    
    # Optimize for M1 Mac
    if device.type == "cpu" or device.type == "mps":
        num_workers = min(num_workers, 2)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable for M1/CPU
    )
    
    # Setup optimizer (lower LR for fine-tuning)
    learning_rate = training_config.get("learning_rate", 1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    # Learning rate scheduler
    num_epochs = training_config.get("num_epochs", 5)
    total_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1
    )
    
    # Create output directory
    output_dir = training_config.get("output_dir", "./models/instruction_tuned")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting instruction tuning for {num_epochs} epochs...")
    accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device, accumulation_steps)
        scheduler.step()
        
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % training_config.get("save_every", 1) == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size
    }, final_model_path)
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nInstruction tuning complete! Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Instruction tune pretrained model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/instruction_tune_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    instruction_tune(args.config)


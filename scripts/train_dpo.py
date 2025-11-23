"""
DPO (Direct Preference Optimization) training for conversational alignment.
Stage 3: Improves politeness, coherence, and helpfulness.
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import ChartLanguageModel


class PreferenceDataset(Dataset):
    """Dataset for DPO training with preference pairs."""
    
    def __init__(self, preferences: list, tokenizer, max_length: int = 512):
        self.preferences = preferences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        prompt = pref.get("prompt", "")
        chosen = pref.get("chosen", "")
        rejected = pref.get("rejected", "")
        
        # Format: prompt + response
        chosen_text = f"{prompt}\n\nAssistant: {chosen}"
        rejected_text = f"{prompt}\n\nAssistant: {rejected}"
        
        # Tokenize
        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(0),
            "prompt": prompt
        }


def dpo_loss(model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta=0.1):
    """
    Compute DPO loss.
    
    DPO loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))
    """
    # Get logits
    chosen_logits, _ = model(input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_logits, _ = model(input_ids=rejected_ids, attention_mask=rejected_mask)
    
    # Compute log probabilities (log_softmax)
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    
    # Get log probs for actual tokens (shift by 1 for next-token prediction)
    batch_size, seq_len = chosen_ids.shape
    chosen_log_probs_selected = chosen_log_probs[:, :-1, :].gather(
        2, chosen_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    rejected_log_probs_selected = rejected_log_probs[:, :-1, :].gather(
        2, rejected_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out padding
    chosen_mask_shifted = chosen_mask[:, 1:].float()
    rejected_mask_shifted = rejected_mask[:, 1:].float()
    
    # Sum log probs (average would also work)
    chosen_log_prob = (chosen_log_probs_selected * chosen_mask_shifted).sum(dim=1)
    rejected_log_prob = (rejected_log_probs_selected * rejected_mask_shifted).sum(dim=1)
    
    # DPO loss
    log_diff = beta * (chosen_log_prob - rejected_log_prob)
    loss = -F.logsigmoid(log_diff).mean()
    
    return loss, chosen_log_prob.mean().item(), rejected_log_prob.mean().item()


def train_dpo(config_path: str = "./configs/dpo_config.yaml"):
    """Main DPO training function."""
    print("=" * 60)
    print("DPO Training - Conversational Alignment")
    print("=" * 60)
    
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
    
    # Load instruction-tuned model
    model_path = model_config["instruction_tuned_model_path"]
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    print(f"Model loaded with {model.get_num_params():,} parameters")
    
    # Load preference data
    pref_path = data_config.get("preference_data_path")
    if not os.path.exists(pref_path):
        print(f"Warning: {pref_path} not found. Creating sample preferences...")
        # Create sample preferences
        preferences = [
            {
                "prompt": "What is machine learning?",
                "chosen": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "rejected": "I don't know."
            }
        ] * 100
        os.makedirs(os.path.dirname(pref_path), exist_ok=True)
        with open(pref_path, 'w') as f:
            json.dump(preferences, f, indent=2)
    
    with open(pref_path, 'r') as f:
        preferences = json.load(f)
    
    print(f"Loaded {len(preferences)} preference pairs")
    
    # Create dataset
    dataset = PreferenceDataset(preferences, tokenizer, max_length=model_config.get("max_seq_len", 512))
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=training_config.get("num_workers", 4)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-6),
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    # Output directory
    output_dir = training_config.get("output_dir", "./models/dpo_tuned")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    num_epochs = training_config.get("num_epochs", 3)
    beta = training_config.get("beta", 0.1)
    accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    
    print(f"\nStarting DPO training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            
            # Compute DPO loss
            loss, chosen_logp, rejected_logp = dpo_loss(
                model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta
            )
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            progress_bar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "chosen_logp": f"{chosen_logp:.2f}",
                "rejected_logp": f"{rejected_logp:.2f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
            "config": config
        }, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size
    }, final_path)
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nDPO training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with DPO")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/dpo_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train_dpo(args.config)


"""
Pretrain the language model from scratch on raw text data.
This is the first stage of training where the model learns language structure.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import ChartLanguageModel


class TextDataset(Dataset):
    """Dataset for pretraining on raw text."""
    
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


def load_text_data(data_dir: str, tokenizer, max_samples: int = None):
    """Load and prepare text data from various sources."""
    all_texts = []
    
    # Load from chat data
    chat_dir = os.path.join(data_dir, "chat")
    if os.path.exists(chat_dir):
        for file in Path(chat_dir).rglob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                for item in data:
                    if "messages" in item:
                        # Combine messages into conversation
                        conversation = ""
                        for msg in item["messages"]:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            conversation += f"{role.capitalize()}: {content}\n"
                        all_texts.append(conversation)
                    elif "text" in item:
                        all_texts.append(item["text"])
    
    # Load from log data
    logs_dir = os.path.join(data_dir, "logs")
    if os.path.exists(logs_dir):
        for file in Path(logs_dir).rglob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                for item in data:
                    # Combine log entry, question, and answer
                    text = f"Log: {item.get('log_entry', '')}\n"
                    text += f"Question: {item.get('question', '')}\n"
                    text += f"Answer: {item.get('answer', '')}\n"
                    all_texts.append(text)
    
    # Load from metadata
    metadata_dir = os.path.join(data_dir, "metadata")
    if os.path.exists(metadata_dir):
        for file in Path(metadata_dir).rglob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                # Convert to text representation
                text = json.dumps(data, indent=2)
                all_texts.append(text)
    
    # Load plain text files
    for ext in ["*.txt", "*.text"]:
        for file in Path(data_dir).rglob(ext):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks if very long
                if len(content) > 10000:
                    chunks = [content[i:i+10000] for i in range(0, len(content), 10000)]
                    all_texts.extend(chunks)
                else:
                    all_texts.append(content)
    
    if max_samples:
        all_texts = all_texts[:max_samples]
    
    print(f"Loaded {len(all_texts)} text samples")
    return all_texts


def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1, 
                scaler=None, max_grad_norm=1.0, global_step=0, warmup_steps=0, base_lr=None, label_smoothing=0.0):
    """
    Train for one epoch with improvements:
    - Mixed precision training (if scaler provided)
    - Gradient clipping
    - Learning rate warmup
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # Determine if we should use mixed precision
    use_amp = scaler is not None
    
    # Determine device type for autocast
    if device.type == "cuda":
        device_type = "cuda"
        dtype = torch.float16
    elif device.type == "mps":
        device_type = "cpu"  # MPS doesn't support autocast, but we can still use scaler
        dtype = torch.float32
    else:
        device_type = "cpu"
        dtype = torch.float32
    
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass with mixed precision if available
            if use_amp and device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, label_smoothing=label_smoothing)
            else:
                logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, label_smoothing=label_smoothing)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            if use_amp and device_type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping (ChatGPT recommendation)
                if use_amp and device_type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                # Learning rate warmup
                # Note: We'll set the base LR in optimizer, warmup will scale from 0 to base_lr
                if use_warmup and warmup_steps > 0 and global_step <= warmup_steps and base_lr is not None:
                    warmup_lr = base_lr * (global_step / warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
            
            total_loss += loss.item() * accumulation_steps
            
            # Calculate perplexity (ChatGPT recommendation)
            perplexity = torch.exp(torch.tensor(loss.item() * accumulation_steps))
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "ppl": f"{perplexity:.1f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        except RuntimeError as e:
            if "out of memory" in str(e) or "MPS" in str(e):
                print(f"\n⚠️  Error at step {step}: {e}")
                print("Clearing cache and continuing...")
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                if scaler:
                    scaler.update()
                continue
            else:
                raise
        except Exception as e:
            print(f"\n❌ Unexpected error at step {step}: {e}")
            raise
    
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, avg_perplexity.item(), global_step


def validate_epoch(model, dataloader, device):
    """
    Validate model on validation set.
    Returns average loss and perplexity.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, avg_perplexity.item()


def pretrain(config_path: str = "./configs/pretrain_config.yaml"):
    """Main pretraining function."""
    print("=" * 50)
    print("Pretraining Language Model from Scratch")
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
    
    # Check if we should use CPU instead (for stability)
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
    
    # Initialize model
    print("Initializing model...")
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
    
    model = model.to(device)
    num_params = model.get_num_params()
    print(f"Model initialized with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Load data
    print("Loading training data...")
    pretraining_path = data_config.get("pretraining_data_path")
    
    if pretraining_path and os.path.exists(pretraining_path):
        # Load from collected pretraining file
        print(f"Loading from {pretraining_path}...")
        with open(pretraining_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Split into chunks
            chunk_size = 10000
            texts = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    else:
        # Fallback to old method
        texts = load_text_data(
            data_config.get("data_dir", "./data"),
            tokenizer,
            max_samples=data_config.get("max_samples")
        )
    
    if len(texts) == 0:
        print("Error: No training data found!")
        return
    
    # Create dataset and split into train/validation (ChatGPT recommendation)
    max_seq_len = model_config.get("max_seq_len", 512)
    full_dataset = TextDataset(texts, tokenizer, max_length=max_seq_len)
    
    # Split dataset: 90% train, 10% validation (ChatGPT recommendation)
    val_split = training_config.get("val_split", 0.1)  # 10% validation
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    print(f"📊 Dataset split: {train_size} train, {val_size} validation ({val_split*100:.1f}% validation)")
    
    # Optimize for M1 Mac
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 4)
    pin_memory = training_config.get("pin_memory", False)
    
    # Reduce workers for M1 (too many can cause issues)
    if device.type == "cpu" or device.type == "mps":
        num_workers = min(num_workers, 2)
        pin_memory = False  # Disable pin_memory for CPU/MPS
    
    # Enable pin_memory for CUDA if configured
    if device.type == "cuda" and pin_memory:
        pin_memory = True
    else:
        pin_memory = False
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Setup optimizer
    learning_rate = training_config.get("learning_rate", 3e-4)
    # Ensure learning_rate is a float
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.01)
    )
    
    # Learning rate schedulers
    num_epochs = training_config.get("num_epochs", 10)
    total_steps = len(train_dataloader) * num_epochs
    
    # Get warmup steps from config, or calculate as 10% of total if not specified
    use_warmup = training_config.get("use_warmup", True)
    if use_warmup:
        warmup_steps = training_config.get("warmup_steps", None)
        if warmup_steps is None:
            warmup_steps = int(total_steps * 0.1)  # Default: 10% warmup
        else:
            warmup_steps = int(warmup_steps)
    else:
        warmup_steps = 0
    
    # Learning rate scheduler configuration
    use_warm_restarts = training_config.get("use_warm_restarts", True)
    use_cosine_decay = training_config.get("use_cosine_decay", False)
    cosine_scheduler = None
    plateau_scheduler = None
    scheduler = None
    
    # Use cosine decay if configured
    if use_cosine_decay:
        # Use CosineAnnealingWarmRestarts if configured
        if use_warm_restarts:
            cosine_restart_epochs = training_config.get("cosine_restart_epochs", 10)
            restart_steps = len(train_dataloader) * cosine_restart_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=restart_steps,  # Restart every N epochs (in steps)
                T_mult=2,  # Double period each restart
                eta_min=learning_rate * 0.1
            )
            print(f"✅ Using CosineAnnealingWarmRestarts (restart every {cosine_restart_epochs} epochs)")
        else:
            # Use standard CosineAnnealingLR
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps if warmup_steps > 0 else total_steps,
                eta_min=learning_rate * 0.1
            )
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=learning_rate * 0.01
            )
            print("✅ Using CosineAnnealingLR + ReduceLROnPlateau")
    elif use_warm_restarts:
        # Fallback: use warm restarts even if cosine_decay not explicitly set
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_dataloader) * 10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=learning_rate * 0.1
        )
        print("✅ Using CosineAnnealingWarmRestarts (legacy mode)")
    else:
        print("⚠️  No learning rate scheduler configured")
    
    # Mixed precision training (ChatGPT recommendation)
    use_mixed_precision = training_config.get("use_mixed_precision", True)
    scaler = None
    if use_mixed_precision and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("✅ Mixed precision training enabled (FP16)")
    elif use_mixed_precision and device.type == "mps":
        # MPS doesn't support autocast, but we can still use scaler for stability
        print("⚠️  Mixed precision requested but MPS doesn't support autocast")
        print("   Continuing with FP32 (MPS is already optimized)")
    
    # Gradient clipping (ChatGPT recommendation)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    print(f"✅ Gradient clipping enabled (max_norm={max_grad_norm})")
    
    # Label smoothing
    label_smoothing = training_config.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        print(f"✅ Label smoothing enabled (smoothing={label_smoothing})")
    
    # Learning rate warmup
    if use_warmup and warmup_steps > 0:
        print(f"✅ Learning rate warmup enabled ({warmup_steps} steps, {warmup_steps/total_steps*100:.1f}% of total)")
    else:
        print("⚠️  Learning rate warmup disabled")
    
    # Early stopping (ChatGPT recommendation)
    early_stopping_patience = training_config.get("early_stopping_patience", 10)
    early_stopping_enabled = training_config.get("early_stopping", True)
    target_loss = training_config.get("target_loss", None)  # Target loss to achieve
    best_val_loss = float('inf')
    patience_counter = 0
    if early_stopping_enabled:
        print(f"✅ Early stopping enabled (patience={early_stopping_patience} epochs)")
    if target_loss:
        print(f"🎯 Target loss: {target_loss:.2f} - Training will stop when validation loss reaches this")
    
    # TensorBoard logging (ChatGPT recommendation)
    use_tensorboard = training_config.get("use_tensorboard", True)
    writer = None
    if use_tensorboard:
        log_dir = training_config.get("tensorboard_log_dir", "./runs/pretraining")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"✅ TensorBoard logging enabled (log_dir={log_dir})")
        print(f"   View with: tensorboard --logdir {log_dir}")
    
    # Create output directory
    output_dir = training_config.get("output_dir", "./models/pretrained")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for resume from checkpoint (automatic for nightly training)
    start_epoch = 0
    checkpoint_path = training_config.get("checkpoint_path")
    
    # Auto-find latest checkpoint if not specified
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        checkpoints = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                    try:
                        epoch_num = int(file.split("_")[-1].split(".")[0])
                        checkpoints.append((epoch_num, os.path.join(output_dir, file)))
                    except:
                        pass
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            checkpoint_path = checkpoints[0][1]
            print(f"📂 Auto-found latest checkpoint: {os.path.basename(checkpoint_path)}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"✅ Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs (starting from epoch {start_epoch})...")
    accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    
    # Track global step for warmup
    global_step = start_epoch * len(train_dataloader)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            try:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                train_loss, train_perplexity, global_step = train_epoch(
                    model, train_dataloader, optimizer, device, accumulation_steps,
                    scaler=scaler, max_grad_norm=max_grad_norm,
                    global_step=global_step, warmup_steps=warmup_steps,
                    base_lr=learning_rate, label_smoothing=label_smoothing
                )
                
                # Validation phase (ChatGPT recommendation)
                val_loss, val_perplexity = validate_epoch(model, val_dataloader, device)
                
                # Update schedulers (only after warmup completes)
                if global_step > warmup_steps:
                    if scheduler is not None:
                        # CosineAnnealingWarmRestarts (step after each epoch)
                        scheduler.step()
                    elif cosine_scheduler is not None:
                        # Original approach
                        cosine_scheduler.step()
                        if plateau_scheduler is not None:
                            plateau_scheduler.step(val_loss)  # Use validation loss for plateau
                
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train PPL: {train_perplexity:.2f} | Val PPL: {val_perplexity:.2f}")
                print(f"Learning rate: {current_lr:.6f}")
                
                # TensorBoard logging (ChatGPT recommendation)
                if writer:
                    writer.add_scalar("Loss/Train", train_loss, epoch + 1)
                    writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
                    writer.add_scalar("Perplexity/Train", train_perplexity, epoch + 1)
                    writer.add_scalar("Perplexity/Validation", val_perplexity, epoch + 1)
                    writer.add_scalar("LearningRate", current_lr, epoch + 1)
                
                # Early stopping (ChatGPT recommendation)
                if early_stopping_enabled:
                    if val_loss < best_val_loss - 0.001:  # Improvement threshold
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        best_model_path = os.path.join(output_dir, "best_model.pt")
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "config": config
                        }, best_model_path)
                        print(f"✅ New best validation loss: {val_loss:.4f} (saved to best_model.pt)")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"\n🛑 Early stopping triggered after {patience_counter} epochs without improvement")
                            print(f"   Best validation loss: {best_val_loss:.4f}")
                            break
                
                # Check if target loss is achieved
                if target_loss and val_loss <= target_loss:
                    print(f"\n🎯 Target loss achieved! Validation loss: {val_loss:.4f} <= {target_loss:.2f}")
                    print(f"✅ Training complete! Model saved to {output_dir}")
                    # Save final model
                    final_model_path = os.path.join(output_dir, "model.pt")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": config
                    }, final_model_path)
                    break
                
                # Save checkpoint
                if (epoch + 1) % training_config.get("save_every", 1) == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    try:
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "config": config
                        }, checkpoint_path)
                        print(f"✅ Checkpoint saved to {checkpoint_path}")
                        
                        # Cleanup old checkpoints to save disk space (ChatGPT recommendation)
                        cleanup_enabled = training_config.get("cleanup_checkpoints", True)
                        if cleanup_enabled:
                            keep_last = training_config.get("keep_last_checkpoints", 5)
                            keep_milestones = training_config.get("keep_milestone_checkpoints", True)
                            milestone_interval = training_config.get("milestone_interval", 10)
                            keep_best = training_config.get("keep_best_checkpoint", True)
                            
                            # Import cleanup function
                            try:
                                import sys
                                cleanup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "checkpoint_cleanup.py")
                                if os.path.exists(cleanup_path):
                                    sys.path.insert(0, os.path.dirname(cleanup_path))
                                    from checkpoint_cleanup import cleanup_checkpoints
                                    deleted_count, deleted_size = cleanup_checkpoints(
                                        output_dir,
                                        keep_last=keep_last,
                                        keep_milestones=keep_milestones,
                                        milestone_interval=milestone_interval,
                                        keep_best=keep_best
                                    )
                                    if deleted_count > 0:
                                        print(f"🧹 Cleaned up {deleted_count} old checkpoints (freed {deleted_size / (1024**3):.2f} GB)")
                            except Exception as e:
                                print(f"⚠️  Checkpoint cleanup failed: {e}")
                    except Exception as e:
                        print(f"⚠️  Failed to save checkpoint: {e}")
                        # Try saving to a backup location
                        backup_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}_backup.pt")
                        try:
                            torch.save({
                                "epoch": epoch + 1,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "config": config
                            }, backup_path)
                            print(f"✅ Saved to backup location: {backup_path}")
                        except Exception as e2:
                            print(f"❌ Backup save also failed: {e2}")
            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user. Saving checkpoint...")
                # Save emergency checkpoint
                emergency_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}_interrupted.pt")
                try:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss if 'avg_loss' in locals() else 0.0,
                        "config": config
                    }, emergency_path)
                    print(f"✅ Emergency checkpoint saved: {emergency_path}")
                except Exception as e:
                    print(f"❌ Failed to save emergency checkpoint: {e}")
                raise
            except Exception as e:
                print(f"\n❌ Error during epoch {epoch + 1}: {e}")
                print("Attempting to save checkpoint before exiting...")
                # Try to save what we have
                emergency_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_error.pt")
                try:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss if 'avg_loss' in locals() else 0.0,
                        "config": config,
                        "error": str(e)
                    }, emergency_path)
                    print(f"✅ Error checkpoint saved: {emergency_path}")
                except:
                    print("❌ Could not save error checkpoint")
                raise
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted. Exiting...")
        raise
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"✅ TensorBoard logs saved to {log_dir}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size
    }, final_model_path)
    
    # Also save in a format that can be loaded easily
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training complete! Model saved to {output_dir}")
    if early_stopping_enabled:
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pretrain language model from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/pretrain_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    pretrain(args.config)


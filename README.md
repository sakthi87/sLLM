# ChartsLLM - Train GPT-2 Large from Scratch

A complete framework for training a GPT-2 Large (762M parameters) language model from scratch, targeting English fluency with a loss of ~2.0.

## 🚀 Quick Start

**New to the project?** Start here: **[QUICK_START.md](QUICK_START.md)**

Complete step-by-step guide from virtual environment setup to training completion.

### Quick Command Sequence

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets wikiextractor warcio requests

# 2. Collect 50GB data
python scripts/collect_pretraining_data.py --output_dir ./data/pretraining --target_size_gb 50

# 3. Train tokenizer (GPT-2 Large: vocab_size=50257)
python scripts/train_tokenizer.py --data_dir ./data --output_dir ./tokenizer --vocab_size 50257

# 4. Start training
# Mac (8GB RAM):
python scripts/pretrain.py --config configs/pretrain_config_mac.yaml

# Linux (120GB RAM - GPT-2 Large):
python scripts/pretrain.py --config configs/pretrain_config_linux.yaml
```

## What You're Building

A **GPT-2 Large (762M parameters)** language model that:
- Achieves English fluency (target loss: ~2.0-2.5)
- Can be fine-tuned for domain-specific tasks
- Supports knowledge distillation for smaller models

## Project Structure

```
ChartsLLM/
├── README.md              # This file
├── START_HERE.md          # Simple guide (read this!)
├── models/                # Model architecture
├── scripts/               # Training scripts
├── configs/               # Configuration files
├── data/                  # Training data
├── docs/                  # All documentation (reference)
└── requirements.txt       # Dependencies
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Complete step-by-step training guide ⭐
- **[DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md)** - Detailed data collection guide (50GB)
- **[TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)** - Two-phase training strategy

## Essential Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Check data size
du -sh data/pretraining/

# Monitor training
tensorboard --logdir runs/

# Test trained model
python scripts/inference.py \
  --model_path models/pretrained/model_epoch_10.pt \
  --config configs/pretrain_config_linux.yaml \
  --prompt "Your prompt here"
```

## Model Details

- **Parameters**: 762M (GPT-2 Large)
- **Architecture**: GPT-style transformer
- **Vocabulary**: 50,257 tokens
- **Target Loss**: ~2.0-2.5
- **Training Time**: 2-3 days (Linux 120GB) or 10+ days (Mac 8GB)
- **Data Required**: 50GB general English text

## Configuration Files

- **Mac (8GB RAM)**: `configs/pretrain_config_mac.yaml`
- **Linux (120GB RAM - GPT-2 Large)**: `configs/pretrain_config_linux.yaml`

## Need Help?

See **[QUICK_START.md](QUICK_START.md)** for the complete step-by-step guide!


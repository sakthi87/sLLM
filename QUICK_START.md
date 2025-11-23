# Quick Start Guide - Complete Training Pipeline

Simple step-by-step commands to train GPT-2 Large from scratch.

---

## Step 1: Setup Python Virtual Environment

### 1.1 Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/subhalakshmiraj/Documents/ChartsLLM

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
which python
```

**Expected Output:**
```
/Users/subhalakshmiraj/Documents/ChartsLLM/venv/bin/python
```

---

## Step 2: Install Dependencies

### 2.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 2.2 Install Requirements

```bash
pip install -r requirements.txt
```

### 2.3 Install Additional Data Collection Dependencies

```bash
# For data collection
pip install datasets wikiextractor warcio requests

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.x.x
Transformers: 4.x.x
```

---

## Step 3: Clean Up Existing Data (Optional)

### 3.1 Remove Broken Files

```bash
# Remove broken Wikipedia file
rm -f data/pretraining/wikipedia.parquet

# Clean up empty or broken files
python scripts/cleanup_data.py --execute
```

---

## Step 4: Collect Training Data (50GB Target)

### 4.1 Collect Pretraining Data

**Option A: Automated Collection (Recommended)**

```bash
# Collect data using the script
python scripts/collect_pretraining_data.py \
  --output_dir ./data/pretraining \
  --target_size_gb 50
```

**Option B: Manual Collection (For Strict Environments)**

See `DATA_COLLECTION_PLAN.md` for detailed manual commands.

**Quick Manual Commands:**

```bash
# 1. Wikipedia (10-20 GB)
mkdir -p downloads/wikipedia
cd downloads/wikipedia
wget --continue https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
pip install wikiextractor
wikiextractor --no-templates --no-style --no-links \
  --min_text_length 100 \
  --output wiki_extracted \
  enwiki-latest-pages-articles.xml.bz2
find wiki_extracted -name '*.txt' -type f -exec cat {} \; > ../../data/pretraining/wikipedia_full.txt
cd ../..

# 2. Project Gutenberg (5-10 GB) - See DATA_COLLECTION_PLAN.md
# 3. OpenWebText (5-10 GB) - See DATA_COLLECTION_PLAN.md
# 4. Common Crawl (10-20 GB) - See DATA_COLLECTION_PLAN.md
# 5. C4 Dataset (10-20 GB) - See DATA_COLLECTION_PLAN.md
```

### 4.2 Verify Data Collection

```bash
# Check total data size
du -sh data/pretraining/

# List files
ls -lh data/pretraining/

# Check individual file sizes
du -h data/pretraining/*.txt
```

**Target:** ~50 GB total

---

## Step 5: Train Tokenizer

### 5.1 Train Tokenizer for GPT-2 Large

```bash
# Train tokenizer with vocab_size=50257 (GPT-2 Large standard)
python scripts/train_tokenizer.py \
  --data_dir ./data \
  --output_dir ./tokenizer \
  --vocab_size 50257
```

**Expected Output:**
```
Training tokenizer...
Tokenizer saved to: ./tokenizer
Vocabulary size: 50257
```

### 5.2 Verify Tokenizer

```bash
# Test tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
print(f'Vocab size: {len(tokenizer)}')
print(f'Test: {tokenizer.encode(\"Hello world\")}')
"
```

**Expected Output:**
```
Vocab size: 50257
Test: [1234, 5678]
```

---

## Step 6: Configure Training

### 6.1 Choose Configuration File

**For Mac (8GB RAM):**
```bash
CONFIG_FILE="configs/pretrain_config_mac.yaml"
```

**For Linux (120GB RAM) - GPT-2 Large:**
```bash
CONFIG_FILE="configs/pretrain_config_linux.yaml"
```

### 6.2 Verify Configuration

```bash
# View configuration
cat $CONFIG_FILE
```

---

## Step 7: Start Pretraining

### 7.1 Start Training

**For Mac:**
```bash
python scripts/pretrain.py --config configs/pretrain_config_mac.yaml
```

**For Linux (GPT-2 Large):**
```bash
python scripts/pretrain.py --config configs/pretrain_config_linux.yaml
```

### 7.2 Monitor Training

**In another terminal (while training):**

```bash
# Activate venv
source venv/bin/activate

# Monitor with TensorBoard
tensorboard --logdir runs/

# Or check training logs
tail -f runs/pretraining/training.log
```

**Training Metrics to Watch:**
- **Loss**: Target ~2.0-2.5 for GPT-2 Large
- **Perplexity**: Target ~7-12 (exp(loss))
- **Learning Rate**: Should follow warmup + cosine decay
- **Validation Loss**: Should track training loss

---

## Step 8: Check Training Progress

### 8.1 Check Checkpoints

```bash
# List checkpoints
ls -lh models/pretrained/

# Check latest checkpoint
ls -lt models/pretrained/ | head -5
```

### 8.2 Validate Training

```bash
# Run validation script
python scripts/validate_training.py \
  --checkpoint_dir models/pretrained/ \
  --config $CONFIG_FILE
```

---

## Step 9: Resume Training (If Interrupted)

### 9.1 Resume from Checkpoint

```bash
# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t models/pretrained/*.pt | head -1)
echo "Resuming from: $LATEST_CHECKPOINT"

# Resume training
python scripts/pretrain.py \
  --config $CONFIG_FILE \
  --resume_from $LATEST_CHECKPOINT
```

---

## Step 10: Test Trained Model

### 10.1 Run Inference

```bash
# Test the trained model
python scripts/inference.py \
  --model_path models/pretrained/model_epoch_10.pt \
  --config $CONFIG_FILE \
  --prompt "The quick brown fox"
```

**Expected Output:**
```
Prompt: The quick brown fox
Generated: ... (model continuation)
```

---

## Complete Command Sequence (Copy-Paste Ready)

```bash
# ============================================
# COMPLETE TRAINING PIPELINE
# ============================================

# Step 1: Setup
cd /Users/subhalakshmiraj/Documents/ChartsLLM
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets wikiextractor warcio requests

# Step 2: Clean up
rm -f data/pretraining/wikipedia.parquet
python scripts/cleanup_data.py --execute

# Step 3: Collect data (choose one)
# Option A: Automated
python scripts/collect_pretraining_data.py \
  --output_dir ./data/pretraining \
  --target_size_gb 50

# Option B: Manual (see DATA_COLLECTION_PLAN.md)

# Step 4: Verify data
du -sh data/pretraining/

# Step 5: Train tokenizer
python scripts/train_tokenizer.py \
  --data_dir ./data \
  --output_dir ./tokenizer \
  --vocab_size 50257

# Step 6: Start training
# For Mac:
python scripts/pretrain.py --config configs/pretrain_config_mac.yaml

# For Linux (GPT-2 Large):
python scripts/pretrain.py --config configs/pretrain_config_linux.yaml

# Step 7: Monitor (in another terminal)
source venv/bin/activate
tensorboard --logdir runs/

# Step 8: Test model
python scripts/inference.py \
  --model_path models/pretrained/model_epoch_10.pt \
  --config configs/pretrain_config_linux.yaml \
  --prompt "Hello, how are you?"
```

---

## Troubleshooting

### Issue: Virtual environment not activating

```bash
# Make sure you're in the project directory
cd /Users/subhalakshmiraj/Documents/ChartsLLM

# Try explicit path
source ./venv/bin/activate
```

### Issue: CUDA/GPU not available

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# For Mac MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Issue: Out of memory

```bash
# Reduce batch size in config file
# Edit configs/pretrain_config_mac.yaml or pretrain_config_linux.yaml
# Reduce: batch_size and/or gradient_accumulation_steps
```

### Issue: Data collection too slow

```bash
# Use manual download (see DATA_COLLECTION_PLAN.md)
# Or collect in phases (10GB at a time)
```

### Issue: Training interrupted

```bash
# Resume from latest checkpoint
LATEST=$(ls -t models/pretrained/*.pt | head -1)
python scripts/pretrain.py \
  --config configs/pretrain_config_linux.yaml \
  --resume_from $LATEST
```

---

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup | 10 min | Virtual env + dependencies |
| Data Collection | 4-6 days | 50GB download + processing |
| Tokenizer Training | 1-2 hours | Depends on data size |
| Pretraining (Mac) | 10+ days | 8GB RAM, overnight runs |
| Pretraining (Linux) | 2-3 days | 120GB RAM, continuous |

**Total (Linux):** ~1 week  
**Total (Mac):** ~2-3 weeks

---

## Next Steps After Pretraining

1. **Domain Fine-tuning** (Optional)
   ```bash
   python scripts/finetune_domain.py --config configs/domain_config.yaml
   ```

2. **Instruction Tuning** (Optional)
   ```bash
   python scripts/instruction_tune.py --config configs/instruction_config.yaml
   ```

3. **Knowledge Distillation** (Optional)
   ```bash
   python scripts/train_distillation.py --config configs/distillation_config.yaml
   ```

---

## Summary

✅ **Setup**: Virtual env + dependencies  
✅ **Data**: Collect 50GB pretraining data  
✅ **Tokenizer**: Train with vocab_size=50257  
✅ **Training**: Run pretrain.py with appropriate config  
✅ **Monitor**: Use TensorBoard  
✅ **Test**: Use inference.py  

**Target Loss**: ~2.0-2.5 for GPT-2 Large  
**Target Perplexity**: ~7-12

---

## Quick Reference

```bash
# Activate venv
source venv/bin/activate

# Check data size
du -sh data/pretraining/

# Check training progress
tensorboard --logdir runs/

# Test model
python scripts/inference.py --model_path models/pretrained/model_epoch_10.pt --config configs/pretrain_config_linux.yaml --prompt "Your prompt here"
```


# RedHat Linux Training Timeline
## System Specifications: 128GB RAM, 24 CPU Cores, 1TB SSD

This document provides detailed timelines for training GPT-2 Large (762M parameters) on a RedHat Linux system with high-end hardware specifications.

---

## System Specifications

- **OS**: RedHat Linux
- **RAM**: 128GB
- **CPU**: 24 cores
- **Storage**: 1TB SSD
- **Model**: GPT-2 Large (762M parameters)
- **Target Loss**: ~2.0-2.5
- **Target Data**: 50GB pretraining data

---

## Phase 1: Environment Setup

### Step 1.1: System Preparation
**Time: 15-30 minutes**

```bash
# Update system packages
sudo yum update -y

# Install Python 3.9+
sudo yum install -y python3 python3-pip python3-devel gcc gcc-c++

# Install CUDA (if GPU available)
# Optional: Install CUDA toolkit for GPU acceleration
```

**Activities:**
- System updates
- Python installation
- Development tools setup
- CUDA installation (if GPU available)

---

### Step 1.2: Virtual Environment Setup
**Time: 5-10 minutes**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install datasets wikiextractor warcio requests
```

**Activities:**
- Virtual environment creation
- Dependency installation
- Verification

**Total Phase 1 Time: 20-40 minutes**

---

## Phase 2: Data Collection (50GB Target)

### Step 2.1: Wikipedia (10-20 GB)
**Time: 6-12 hours**

**Download:**
```bash
# Download Wikipedia dump (~20GB compressed)
wget --continue https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# Time: 2-4 hours (depends on internet speed)
```

**Extraction & Processing:**
```bash
# Extract text (24 cores = fast processing)
wikiextractor --no-templates --no-style --no-links \
  --min_text_length 100 \
  --output wiki_extracted \
  --processes 24 \
  enwiki-latest-pages-articles.xml.bz2
# Time: 2-3 hours (with 24 cores)

# Combine files
find wiki_extracted -name '*.txt' -type f -exec cat {} \; > data/pretraining/wikipedia_full.txt
# Time: 30-60 minutes
```

**Expected Output:** 10-20 GB text file

---

### Step 2.2: Project Gutenberg (5-10 GB)
**Time: 2-4 hours**

```bash
# Download 500+ books
python scripts/collect_pretraining_data.py \
  --output_dir ./data/pretraining \
  --gutenberg_books 500
# Time: 2-4 hours (with rate limiting)
```

**Expected Output:** 5-10 GB text file

---

### Step 2.3: OpenWebText (5-10 GB)
**Time: 3-6 hours**

```bash
# Download via HuggingFace (streaming)
python scripts/collect_pretraining_data.py \
  --output_dir ./data/pretraining \
  --openwebtext_samples 1000000
# Time: 3-6 hours (depends on HuggingFace speed)
```

**Expected Output:** 5-10 GB text file

---

### Step 2.4: Common Crawl (10-20 GB)
**Time: 8-16 hours**

```bash
# Download and process WARC files
# See DATA_COLLECTION_PLAN.md for detailed commands
# Time: 6-12 hours download + 2-4 hours processing
```

**Expected Output:** 10-20 GB text file

---

### Step 2.5: C4 Dataset (10-20 GB)
**Time: 4-8 hours**

```bash
# Download via HuggingFace
python scripts/download_c4.py \
  --output_dir ./data/pretraining \
  --max_samples 5000000
# Time: 4-8 hours
```

**Expected Output:** 10-20 GB text file

---

### Step 2.6: Data Verification
**Time: 15-30 minutes**

```bash
# Verify total size
du -sh data/pretraining/
# Should be ~50GB

# Check file integrity
ls -lh data/pretraining/
```

**Total Phase 2 Time: 23-46 hours (1-2 days)**

**Note:** Can be parallelized across multiple terminals/sessions to reduce time.

---

## Phase 3: Tokenizer Training

### Step 3.1: Train Tokenizer
**Time: 2-4 hours**

```bash
# Train tokenizer with vocab_size=50257 (GPT-2 Large)
python scripts/train_tokenizer.py \
  --data_dir ./data \
  --output_dir ./tokenizer \
  --vocab_size 50257 \
  --num_threads 24
```

**Factors:**
- 50GB data processing
- 24 CPU cores (parallel processing)
- Vocabulary size: 50,257 tokens

**Expected Output:** Trained tokenizer in `./tokenizer/`

---

### Step 3.2: Verify Tokenizer
**Time: 5 minutes**

```bash
# Test tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
print(f'Vocab size: {len(tokenizer)}')
"
```

**Total Phase 3 Time: 2-4 hours**

---

## Phase 4: Model Pretraining (GPT-2 Large)

### Step 4.1: Training Configuration
**Time: 5 minutes**

**System Capabilities:**
- **128GB RAM**: Can handle large batch sizes
- **24 CPU cores**: Fast data loading and preprocessing
- **1TB SSD**: Fast I/O for data streaming

**Recommended Config Adjustments:**
```yaml
training:
  batch_size: 24                   # Maximum batch size for 128GB RAM (6x increase from 4)
  gradient_accumulation_steps: 8   # Keep same
  num_epochs: 100
  learning_rate: 3e-4
  num_workers: 24                  # Use all 24 cores for data loading
```

**Effective Batch Size:** 24 × 8 = 192 samples per update

**Memory Usage:**
- Model (FP16): ~1.5 GB
- Optimizer (Adam): ~3.1 GB
- Activations: ~9.1 GB
- Total: ~15.2 GB (11.9% of 128GB)
- Headroom: ~113GB available (88% free!)

---

### Step 4.2: Training Execution
**Time: 36-72 hours (1.5-3 days)**

**Training Metrics:**
- **Model Size**: 762M parameters
- **Sequence Length**: 1024 tokens
- **Data Size**: 50GB (~10B tokens)
- **Steps per Epoch**: ~78,125 (with batch_size=128)
- **Total Steps**: ~7,812,500 (100 epochs)

**Estimated Training Speed:**
- **With batch_size=24 and 24 CPU cores**: ~6-12 steps/second (6x faster)
- **Per epoch**: ~1.5-3 hours (6x faster than batch_size=4)
- **100 epochs**: ~12-24 hours (0.5-1 day) - **6x faster!**

**Progress Milestones:**
- **Epoch 1**: ~12-20 hours (initial setup + first epoch)
- **Epoch 10**: Loss should drop to ~3.5-4.0
- **Epoch 25**: Loss should drop to ~3.0-3.5
- **Epoch 50**: Loss should drop to ~2.5-3.0
- **Epoch 75**: Loss should drop to ~2.2-2.7
- **Epoch 100**: Loss target ~2.0-2.5

---

### Step 4.3: Monitoring & Checkpoints
**Time: Continuous (background)**

```bash
# Monitor training
tensorboard --logdir runs/ --port 6006

# Check checkpoints
ls -lh models/pretrained/

# Validate training
python scripts/validate_training.py \
  --checkpoint_dir models/pretrained/ \
  --config configs/pretrain_config_linux.yaml
```

**Checkpoint Frequency:** Every epoch (saves ~3GB per checkpoint)

---

### Step 4.4: Resume Training (if needed)
**Time: Minimal (automatic)**

```bash
# Resume from latest checkpoint
LATEST=$(ls -t models/pretrained/*.pt | head -1)
python scripts/pretrain.py \
  --config configs/pretrain_config_linux.yaml \
  --resume_from $LATEST
```

**Total Phase 4 Time: 12-24 hours (0.5-1 day)** - **6x faster with maximum batch size!**

---

## Phase 5: Validation & Testing

### Step 5.1: Model Evaluation
**Time: 30-60 minutes**

```bash
# Run validation
python scripts/validate_training.py \
  --checkpoint_dir models/pretrained/ \
  --config configs/pretrain_config_linux.yaml

# Test inference
python scripts/inference.py \
  --model_path models/pretrained/model_epoch_100.pt \
  --config configs/pretrain_config_linux.yaml \
  --prompt "The quick brown fox"
```

**Expected Results:**
- **Loss**: ~2.0-2.5
- **Perplexity**: ~7-12 (exp(loss))
- **Coherent text generation**

---

### Step 5.2: Performance Analysis
**Time: 15-30 minutes**

```bash
# Analyze training metrics
tensorboard --logdir runs/

# Check model size
du -sh models/pretrained/model_epoch_100.pt
# Expected: ~3GB (FP32) or ~1.5GB (FP16)
```

**Total Phase 5 Time: 45-90 minutes**

---

## Complete Timeline Summary

| Phase | Activity | Time | Cumulative |
|-------|----------|------|------------|
| **Phase 1** | Environment Setup | 20-40 min | 40 min |
| **Phase 2** | Data Collection (50GB) | 23-46 hours | 1-2 days |
| **Phase 3** | Tokenizer Training | 2-4 hours | 1-2 days |
| **Phase 4** | Model Pretraining | 12-24 hours | 1.5-2.5 days |
| **Phase 5** | Validation & Testing | 45-90 min | 3-5 days |
| **TOTAL** | **Complete Training** | **~1.5-2.5 days** | **~1.5-2.5 days** |

---

## Optimized Timeline (Parallel Data Collection)

If data collection is parallelized:

| Phase | Activity | Time | Cumulative |
|-------|----------|------|------------|
| **Phase 1** | Environment Setup | 20-40 min | 40 min |
| **Phase 2** | Data Collection (parallel) | 12-18 hours | 1 day |
| **Phase 3** | Tokenizer Training | 2-4 hours | 1 day |
| **Phase 4** | Model Pretraining | 12-24 hours | 1.5-2 days |
| **Phase 5** | Validation & Testing | 45-90 min | 4-5 days |
| **TOTAL** | **Complete Training** | **~1.5-2 days** | **~1.5-2 days** |

---

## Resource Utilization

### Memory Usage
- **Training**: ~15-18GB RAM (with batch_size=24, FP16 mixed precision)
- **Data Loading**: ~5-10GB RAM (24 workers)
- **System**: ~8GB RAM
- **Total**: ~28-36GB (only 22-28% of 128GB - 72-78% headroom available!)

### CPU Usage
- **Training**: 1-2 cores (model forward/backward)
- **Data Loading**: 24 cores (parallel workers)
- **Total**: 24 cores fully utilized

### Disk Usage
- **Data**: 50GB (pretraining data)
- **Checkpoints**: ~3GB × 100 = 300GB (if saving all)
- **Tokenizer**: ~100MB
- **Logs**: ~1-5GB
- **Total**: ~350-400GB (fits in 1TB SSD)

---

## Performance Optimizations

### 1. Increase Batch Size (Maximum Optimized)
```yaml
batch_size: 24  # Maximum setting - uses only 11.9% of 128GB RAM
```

### 2. Use Mixed Precision
```yaml
mixed_precision: true  # FP16 instead of FP32
```
**Benefit:** 2x faster training, 50% less memory

### 3. Optimize Data Loading
```yaml
num_workers: 24  # Use all cores
prefetch_factor: 4  # Prefetch batches
```

### 4. Use Gradient Checkpointing
```yaml
gradient_checkpointing: true  # Trade compute for memory
```

**With Maximum Optimizations (Applied):**
- **Training Time**: 12-24 hours (0.5-1 day) with batch_size=24
- **Total Time**: **1.5-2 days** (optimized from original 4-6 days - **3-4x faster!**)
- **Memory**: Only 11.9% utilization - maximum performance with safety margin

---

## Daily Schedule Example

### Day 1: Setup + Data Collection
- **Morning (9 AM)**: Environment setup (40 min)
- **Rest of Day**: Start data collection (Wikipedia, Gutenberg)
- **Evening**: Continue data collection (OpenWebText, Common Crawl)

### Day 2: Data Collection + Tokenizer
- **Morning**: Complete data collection (C4 dataset)
- **Afternoon**: Verify data (30 min) + Train tokenizer (2-4 hours)
- **Evening**: Start pretraining

### Day 3-4: Pretraining
- **Continuous**: Monitor training, check checkpoints
- **Checkpoints**: Every epoch (~10-20 hours each)

### Day 5: Completion + Validation
- **Morning**: Training completion
- **Afternoon**: Validation and testing
- **Evening**: Analysis and documentation

---

## Monitoring Commands

### Real-time Monitoring
```bash
# Terminal 1: Training
python scripts/pretrain.py --config configs/pretrain_config_linux.yaml

# Terminal 2: TensorBoard
tensorboard --logdir runs/ --port 6006

# Terminal 3: System monitoring
watch -n 5 'free -h && echo && df -h && echo && top -bn1 | head -20'

# Terminal 4: Training logs
tail -f runs/pretraining/training.log
```

### Checkpoint Management
```bash
# List checkpoints
ls -lh models/pretrained/

# Keep only last 10 checkpoints
python scripts/checkpoint_cleanup.py \
  --checkpoint_dir models/pretrained/ \
  --keep_last 10
```

---

## Troubleshooting Timeline

### Issue: Out of Memory
**Time Impact:** +1-2 hours
- Reduce batch_size to 4
- Enable gradient checkpointing
- Reduce num_workers to 12

### Issue: Training Interrupted
**Time Impact:** Minimal (resume from checkpoint)
- Automatic resume from latest checkpoint
- No data loss

### Issue: Slow Data Loading
**Time Impact:** +10-20% training time
- Increase num_workers
- Use SSD (already have)
- Pre-process data into chunks

---

## Success Criteria

### Training Complete When:
- ✅ Loss reaches ~2.0-2.5
- ✅ Perplexity reaches ~7-12
- ✅ Validation loss tracks training loss
- ✅ Generated text is coherent
- ✅ 100 epochs completed (or early stopping)

### Expected Final Metrics:
- **Training Loss**: 2.0-2.5
- **Validation Loss**: 2.1-2.6
- **Perplexity**: 7-12
- **Model Size**: ~3GB (FP32) or ~1.5GB (FP16)

---

## Next Steps After Pretraining

### Optional: Domain Fine-tuning
**Time: 12-24 hours**
```bash
python scripts/finetune_domain.py --config configs/domain_config.yaml
```

### Optional: Instruction Tuning
**Time: 6-12 hours**
```bash
python scripts/instruction_tune.py --config configs/instruction_config.yaml
```

### Optional: Knowledge Distillation
**Time: 24-48 hours**
```bash
python scripts/train_distillation.py --config configs/distillation_config.yaml
```

---

## Summary

**Total Training Time: 1.5-2 days** (with maximum batch_size=24)

**Breakdown:**
- Setup: 40 minutes
- Data Collection: 1-2 days (can be parallelized)
- Tokenizer: 2-4 hours
- Pretraining: 1.5-3 days (can be optimized to 1-2 days)
- Validation: 1 hour

**System Utilization:**
- RAM: ~28-36GB/128GB (22-28% utilization) - **72-78% headroom available!**
- CPU: 24/24 cores (100% utilization)
- Disk: ~400GB/1TB (40% utilization)

**Expected Results:**
- GPT-2 Large model (762M parameters)
- Loss: ~2.0-2.5
- Perplexity: ~7-12
- English fluency achieved

---

## Quick Reference

```bash
# Start training
python scripts/pretrain.py --config configs/pretrain_config_linux.yaml

# Monitor
tensorboard --logdir runs/

# Check progress
ls -lh models/pretrained/

# Test model
python scripts/inference.py \
  --model_path models/pretrained/model_epoch_100.pt \
  --config configs/pretrain_config_linux.yaml \
  --prompt "Your prompt here"
```


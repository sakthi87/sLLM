# Data Collection Plan for 50GB Pretraining Data

## Quick Reference

**For strict environments, jump to:** [Manual Download Commands](#manual-download-commands-for-strict-environments)

**Quick Start (Manual):**
```bash
# 1. Wikipedia (10-20 GB)
wget --continue https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o wiki_output/
find wiki_output/ -name '*.txt' -exec cat {} \; > data/pretraining/wikipedia_full.txt

# 2. Project Gutenberg (5-10 GB) - See manual commands section

# 3. OpenWebText (5-10 GB) - See manual commands section

# 4. Common Crawl (10-20 GB) - See manual commands section

# 5. C4 Dataset (10-20 GB) - See manual commands section
```

---

## Current Status

### Current Data in `pretraining/`:
- `combined_pretraining_data.txt`: 60.10 MB
- `gutenberg.txt`: 34.57 MB
- `chart_descriptions.txt`: 1.09 MB
- `wikipedia.parquet`: 15 bytes (broken - remove)

**Total: 95.75 MB (0.0935 GB)**

### Target vs Current:
- **Target**: 50 GB
- **Current**: 0.0935 GB
- **Needed**: 49.91 GB
- **Progress**: 0.19%

---

## Data Sources to Collect

### 1. Wikipedia (10-20 GB) - Priority 1

**Source**: https://dumps.wikimedia.org/enwiki/latest/

**File**: `enwiki-latest-pages-articles.xml.bz2` (~20GB compressed)

**Method (Automated)**:
```bash
# Install wikiextractor
pip install wikiextractor

# Download Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Extract text
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o wiki_output/

# Combine and clean
find wiki_output/ -name '*.txt' -exec cat {} \; > data/pretraining/wikipedia_full.txt
```

**Method (Manual/Strict Environment)**: See [Manual Download Commands - Wikipedia](#1-wikipedia---manual-download-10-20-gb)

**Expected**: 10-20 GB after processing

---

### 2. Project Gutenberg (5-10 GB) - Priority 2

**Source**: https://www.gutenberg.org/

**Method (Automated)**: 
- Your script already supports this (`download_project_gutenberg`)
- Scale up from 50 books to 500+ books
- Run: `python scripts/collect_pretraining_data.py`

**Method (Manual/Strict Environment)**: See [Manual Download Commands - Project Gutenberg](#2-project-gutenberg---manual-download-5-10-gb)

**Expected**: 5-10 GB

**Enhancement needed**:
- Update `num_books` parameter from 50 to 500+
- Add more book IDs to the list

---

### 3. OpenWebText (5-10 GB) - Priority 3

**Source**: HuggingFace datasets

**Method (Automated)**:
```python
from datasets import load_dataset

dataset = load_dataset("openwebtext", split="train", streaming=True)

with open("data/pretraining/openwebtext.txt", "w") as f:
    for i, example in enumerate(dataset):
        if i >= 1_000_000:  # Scale up from 100K
            break
        text = example.get('text', '')
        if text and len(text) > 100:
            f.write(text + '\n\n')
```

**Method (Manual/Strict Environment)**: See [Manual Download Commands - OpenWebText](#3-openwebtext---manual-download-5-10-gb)

**Expected**: 5-10 GB

**Enhancement needed**:
- Update `download_openwebtext_subset` to collect 1M+ samples instead of 100K

---

### 4. Common Crawl (10-20 GB) - Priority 4

**Source**: https://commoncrawl.org/

**Method (Manual/Strict Environment)**: See [Manual Download Commands - Common Crawl](#4-common-crawl---manual-download-10-20-gb)

**Expected**: 10-20 GB

**Steps**:
1. Download WARC files from Common Crawl
2. Extract text using `warcio` library
3. Filter for English content
4. Clean and process

**Script needed**: Add Common Crawl download function

---

### 5. C4 Dataset (10-20 GB) - Priority 5

**Source**: HuggingFace datasets

**Method (Automated)**:
```python
from datasets import load_dataset

dataset = load_dataset("c4", "en", streaming=True)

with open("data/pretraining/c4.txt", "w") as f:
    for i, example in enumerate(dataset):
        if i >= 5_000_000:  # Adjust based on size needed
            break
        text = example.get('text', '')
        if text:
            f.write(text + '\n\n')
```

**Method (Manual/Strict Environment)**: See [Manual Download Commands - C4 Dataset](#5-c4-dataset---manual-download-10-20-gb)

**Expected**: 10-20 GB

**Script needed**: Add C4 download function

---

## Manual Download Commands (For Strict Environments)

This section provides manual commands that can be run directly in strict/restricted environments without relying on Python scripts or interactive prompts.

### Prerequisites

```bash
# Install required tools (if not available)
# For Debian/Ubuntu:
sudo apt-get update
sudo apt-get install -y wget curl bzip2 python3 python3-pip

# For macOS:
brew install wget bzip2

# Install Python packages
pip3 install --no-input wikiextractor warcio datasets
```

---

### 1. Wikipedia - Manual Download (10-20 GB)

#### Step 1: Download Wikipedia Dump

```bash
# Create download directory
mkdir -p downloads/wikipedia
cd downloads/wikipedia

# Download Wikipedia dump (non-interactive, resume support)
wget --continue --no-check-certificate \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Verify download
ls -lh enwiki-latest-pages-articles.xml.bz2
```

**Alternative using curl:**
```bash
curl -L -C - -o enwiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

#### Step 2: Extract and Process

```bash
# Install wikiextractor (if not installed)
pip3 install --no-input wikiextractor

# Extract text from Wikipedia dump
wikiextractor --no-templates --no-style --no-links \
  --min_text_length 100 \
  --output wiki_extracted \
  enwiki-latest-pages-articles.xml.bz2

# Combine all extracted text files
find wiki_extracted -name '*.txt' -type f -exec cat {} \; > ../../data/pretraining/wikipedia_full.txt

# Clean up (optional)
cd ../..
rm -rf downloads/wikipedia/wiki_extracted
```

#### Step 3: Verify

```bash
# Check file size
du -sh data/pretraining/wikipedia_full.txt

# Check line count
wc -l data/pretraining/wikipedia_full.txt
```

---

### 2. Project Gutenberg - Manual Download (5-10 GB)

#### Step 1: Download Book List

```bash
# Create directory
mkdir -p downloads/gutenberg
cd downloads/gutenberg

# Download catalog (to get book IDs)
wget --no-check-certificate \
  https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv

# Extract popular book IDs (first 500)
head -n 501 pg_catalog.csv | tail -n 500 | cut -d',' -f1 | grep -E '^[0-9]+$' > book_ids.txt
```

#### Step 2: Download Books (Bulk)

```bash
# Download books in batch (non-interactive)
BASE_URL="https://www.gutenberg.org/files"
OUTPUT_FILE="../../data/pretraining/gutenberg_full.txt"

# Clear output file
> "$OUTPUT_FILE"

# Download and process books
while IFS= read -r book_id; do
    # Try different file formats
    for ext in "txt" "txt.utf-8"; do
        URL="${BASE_URL}/${book_id}/${book_id}-${ext}"
        if wget --quiet --spider "$URL" 2>/dev/null; then
            echo "Downloading book $book_id..."
            wget --quiet --no-check-certificate -O - "$URL" | \
                sed -e '/^[[:space:]]*$/d' \
                    -e '/^[[:space:]]*START OF THIS PROJECT GUTENBERG/d' \
                    -e '/^[[:space:]]*END OF THIS PROJECT GUTENBERG/d' \
                    -e '/^[[:space:]]*Project Gutenberg/d' \
                    -e '/^[[:space:]]*This eBook/d' >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            break
        fi
    done
    # Rate limiting (optional)
    sleep 0.5
done < book_ids.txt

cd ../..
```

**Alternative: Download specific popular books**

```bash
# List of popular Gutenberg book IDs (first 500)
BOOK_IDS=(1342 11 84 2701 98 74 76 158 160 161 163 166 172 174 195 259 260 263 264 270 275 278 302 320 345 408 430 520 730 768 821 880 1232 1322 1400 145 1524 16328 174 1787 1800 1952 2148 215 219 225 244 2554 2591 2600 2701 2800 30254 3200 345 408 430 520 730 768 821 880 1232 1322 1400 145 1524 16328 174 1787 1800 1952 2148 215 219 225 244 2554 2591 2600 2701 2800)

BASE_URL="https://www.gutenberg.org/files"
OUTPUT_FILE="data/pretraining/gutenberg_full.txt"

> "$OUTPUT_FILE"

for book_id in "${BOOK_IDS[@]}"; do
    for ext in "txt" "txt.utf-8"; do
        URL="${BASE_URL}/${book_id}/${book_id}-${ext}"
        if wget --quiet --spider "$URL" 2>/dev/null; then
            wget --quiet --no-check-certificate -O - "$URL" | \
                sed -e '/^[[:space:]]*$/d' \
                    -e '/START OF THIS PROJECT GUTENBERG/d' \
                    -e '/END OF THIS PROJECT GUTENBERG/d' >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            break
        fi
    done
    sleep 0.5
done
```

#### Step 3: Verify

```bash
du -sh data/pretraining/gutenberg_full.txt
wc -l data/pretraining/gutenberg_full.txt
```

---

### 3. OpenWebText - Manual Download (5-10 GB)

#### Method 1: Using HuggingFace Datasets (Python Script)

Create a standalone script `download_openwebtext.py`:

```bash
cat > download_openwebtext.py << 'PYEOF'
#!/usr/bin/env python3
import sys
from datasets import load_dataset

output_file = "data/pretraining/openwebtext.txt"
max_samples = 1_000_000

print(f"Downloading OpenWebText (max {max_samples} samples)...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

with open(output_file, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        if i % 10000 == 0:
            print(f"Processed {i} samples...")
        text = example.get('text', '')
        if text and len(text) > 100:
            f.write(text + '\n\n')

print(f"Done! Saved to {output_file}")
PYEOF

chmod +x download_openwebtext.py
python3 download_openwebtext.py
```

#### Method 2: Direct Download from HuggingFace (if available)

```bash
# Check if direct download is available
# Note: OpenWebText is large, streaming is recommended
# Use Method 1 above
```

---

### 4. Common Crawl - Manual Download (10-20 GB)

#### Step 1: Get Latest Crawl Index

```bash
mkdir -p downloads/commoncrawl
cd downloads/commoncrawl

# Get latest crawl index
wget --no-check-certificate \
  https://commoncrawl.org/robots.txt

# Extract latest crawl (example: CC-MAIN-2023-23)
LATEST_CRAWL="CC-MAIN-2023-23"  # Update with latest
```

#### Step 2: Download WARC Files

```bash
# Get list of WARC files for English content
wget --no-check-certificate \
  "https://data.commoncrawl.org/crawl-data/${LATEST_CRAWL}/warc.paths.gz"

# Extract paths
gunzip warc.paths.gz

# Download first 10-20 WARC files (each ~1GB)
head -n 20 warc.paths | while read warc_path; do
    warc_file=$(basename "$warc_path")
    echo "Downloading $warc_file..."
    wget --continue --no-check-certificate \
      "https://data.commoncrawl.org/$warc_path" \
      -O "$warc_file"
done
```

#### Step 3: Extract Text from WARC Files

```bash
# Install warcio
pip3 install --no-input warcio

# Extract text from WARC files
OUTPUT_FILE="../../data/pretraining/commoncrawl.txt"
> "$OUTPUT_FILE"

for warc_file in *.warc.gz; do
    echo "Processing $warc_file..."
    python3 << PYEOF
import warcio
import sys

with open("$warc_file", 'rb') as stream:
    for record in warcio.ArchiveIterator(stream):
        if record.rec_type == 'response':
            content = record.content_stream().read()
            # Simple text extraction (improve as needed)
            text = content.decode('utf-8', errors='ignore')
            # Filter for English, clean text
            if len(text) > 500:
                with open("$OUTPUT_FILE", "a") as f:
                    f.write(text + "\n\n")
PYEOF
done

cd ../..
```

---

### 5. C4 Dataset - Manual Download (10-20 GB)

#### Method: Using HuggingFace Datasets (Python Script)

Create a standalone script `download_c4.py`:

```bash
cat > download_c4.py << 'PYEOF'
#!/usr/bin/env python3
from datasets import load_dataset

output_file = "data/pretraining/c4.txt"
max_samples = 5_000_000  # Adjust based on size needed

print(f"Downloading C4 dataset (max {max_samples} samples)...")
dataset = load_dataset("c4", "en", streaming=True, split="train")

with open(output_file, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        if i % 50000 == 0:
            print(f"Processed {i} samples...")
        text = example.get('text', '')
        if text and len(text) > 200:  # Filter short texts
            f.write(text + '\n\n')

print(f"Done! Saved to {output_file}")
PYEOF

chmod +x download_c4.py
python3 download_c4.py
```

---

### Complete Manual Collection Script

Create a master script `manual_collect_all.sh`:

```bash
cat > manual_collect_all.sh << 'EOF'
#!/bin/bash
set -e  # Exit on error

# Configuration
TARGET_SIZE_GB=50
OUTPUT_DIR="data/pretraining"
DOWNLOAD_DIR="downloads"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo "=========================================="
echo "Manual Data Collection for 50GB"
echo "=========================================="
echo ""

# 1. Wikipedia (10-20 GB)
echo "Step 1: Downloading Wikipedia..."
cd "$DOWNLOAD_DIR"
mkdir -p wikipedia && cd wikipedia
wget --continue --no-check-certificate \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
pip3 install --no-input wikiextractor
wikiextractor --no-templates --no-style --no-links \
  --min_text_length 100 \
  --output wiki_extracted \
  enwiki-latest-pages-articles.xml.bz2
find wiki_extracted -name '*.txt' -type f -exec cat {} \; > ../../"$OUTPUT_DIR"/wikipedia_full.txt
cd ../..

# 2. Project Gutenberg (5-10 GB)
echo "Step 2: Downloading Project Gutenberg..."
# (Use the batch download commands from section 2 above)
# ... (add Gutenberg download commands)

# 3. OpenWebText (5-10 GB)
echo "Step 3: Downloading OpenWebText..."
python3 download_openwebtext.py

# 4. Common Crawl (10-20 GB)
echo "Step 4: Downloading Common Crawl..."
# (Use the Common Crawl commands from section 4 above)
# ... (add Common Crawl download commands)

# 5. C4 Dataset (10-20 GB)
echo "Step 5: Downloading C4 dataset..."
python3 download_c4.py

# Verify total size
echo ""
echo "=========================================="
echo "Collection Complete!"
echo "=========================================="
du -sh "$OUTPUT_DIR"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"
EOF

chmod +x manual_collect_all.sh
```

---

### Running Manual Collection in Strict Environment

```bash
# 1. Set environment variables (if needed)
export DEBIAN_FRONTEND=noninteractive
export PIP_NO_INPUT=1

# 2. Run individual downloads
bash -x manual_collect_all.sh

# 3. Monitor progress
watch -n 60 'du -sh data/pretraining/'

# 4. Check individual file sizes
du -sh data/pretraining/*.txt
```

---

### Troubleshooting for Strict Environments

#### If wget is not available:
```bash
# Use curl instead
curl -L -C - -o filename https://url.com/file
```

#### If pip install fails:
```bash
# Use --no-deps and install manually
pip3 install --no-input --no-deps wikiextractor
# Then install dependencies separately
```

#### If Python scripts fail:
```bash
# Run with explicit Python path
/usr/bin/python3 script.py

# Or set PYTHONPATH
export PYTHONPATH=/usr/lib/python3/dist-packages
```

#### Rate limiting:
```bash
# Add delays between downloads
sleep 1  # Wait 1 second between requests
```

#### Resume interrupted downloads:
```bash
# wget automatically resumes with --continue
wget --continue URL

# curl resumes with -C -
curl -L -C - -o file URL
```

---

## Recommended Collection Strategy

### Phase 1: Quick Collection (20 GB) - Days 1-2

1. **Wikipedia**: 10 GB
   - Download full dump
   - Extract and process
   - Time: 1 day

2. **Project Gutenberg**: 5 GB
   - Scale up to 500+ books
   - Time: 4-6 hours

3. **OpenWebText**: 5 GB
   - Scale up to 1M+ samples
   - Time: 4-6 hours

**Total**: ~20 GB in 1-2 days

---

### Phase 2: Scale Up (20 GB) - Days 3-5

1. **More Wikipedia**: 10 GB
   - Additional processing
   - Different articles/sections

2. **Common Crawl**: 10 GB
   - Download and process WARC files
   - Filter for English

**Total**: ~20 GB in 2-3 days

---

### Phase 3: Final Push (10 GB) - Day 6

1. **C4 Dataset**: 10 GB
   - Download via HuggingFace
   - Process and save

**Total**: ~10 GB in 1 day

---

## Total Timeline

**Estimated Time**: 4-6 days for 50GB collection

**Factors**:
- Internet speed
- Processing power
- Data source availability

---

## Immediate Next Steps

### Step 1: Clean Up
```bash
# Remove broken file
rm data/pretraining/wikipedia.parquet
```

### Step 2: Enhance Collection Script

Update `scripts/collect_pretraining_data.py`:

1. **Wikipedia**: Support full dump download
2. **Project Gutenberg**: Scale to 500+ books
3. **OpenWebText**: Scale to 1M+ samples
4. **Add Common Crawl**: New function
5. **Add C4**: New function

### Step 3: Run Collection
```bash
python scripts/collect_pretraining_data.py \
  --output_dir ./data/pretraining \
  --target_size_gb 50
```

### Step 4: Monitor Progress
```bash
# Check size
du -sh data/pretraining/

# Verify files
ls -lh data/pretraining/
```

---

## Data Quality Checklist

After collection, verify:

- ✅ Text is clean (no excessive HTML/XML)
- ✅ English language only
- ✅ Proper sentence structure
- ✅ No excessive repetition
- ✅ Diverse topics and styles
- ✅ Total size: ~50 GB

---

## Summary

**Current**: 0.0935 GB (0.19% complete)  
**Target**: 50 GB  
**Needed**: 49.91 GB  

**Primary Sources**:
1. Wikipedia (10-20 GB)
2. Project Gutenberg (5-10 GB)
3. OpenWebText (5-10 GB)
4. Common Crawl (10-20 GB)
5. C4 Dataset (10-20 GB)

**Timeline**: 4-6 days

**Next Action**: Enhance collection script and start Phase 1 collection.


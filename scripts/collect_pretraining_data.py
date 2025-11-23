"""
Collect pretraining data from open sources.
Downloads and prepares 10-20 GB of clean English text for pretraining.
"""
import os
import sys
import requests
import json
from pathlib import Path
from tqdm import tqdm
import gzip
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_wikipedia_subset(output_dir: str, max_size_mb: int = 500):
    """
    Download a subset of Wikipedia for pretraining.
    Uses a smaller dump for faster download.
    """
    print("Downloading Wikipedia subset...")
    
    # Use a smaller Wikipedia dump (first 500MB)
    # For full dataset, use: https://dumps.wikimedia.org/enwiki/latest/
    wiki_url = "https://huggingface.co/datasets/wikipedia/resolve/main/20220301.en/20220301.en.parquet"
    
    output_path = os.path.join(output_dir, "wikipedia.parquet")
    
    try:
        download_file(wiki_url, output_path)
        print(f"✓ Wikipedia downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Wikipedia download failed: {e}")
        return None


def download_openwebtext_subset(output_dir: str):
    """
    Download OpenWebText subset.
    """
    print("Downloading OpenWebText subset...")
    
    # Use HuggingFace datasets API
    try:
        from datasets import load_dataset
        
        print("Loading OpenWebText from HuggingFace...")
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        output_path = os.path.join(output_dir, "openwebtext.txt")
        max_samples = 100000  # Limit for initial collection
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, example in enumerate(tqdm(dataset, desc="Collecting OpenWebText")):
                if i >= max_samples:
                    break
                text = example.get('text', '')
                if text and len(text) > 100:  # Filter very short texts
                    f.write(text + '\n\n')
        
        print(f"✓ OpenWebText collected to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ OpenWebText download failed: {e}")
        print("  Install: pip install datasets")
        return None


def download_project_gutenberg(output_dir: str, num_books: int = 100):
    """
    Download books from Project Gutenberg.
    """
    print("Downloading Project Gutenberg books...")
    
    # List of popular Gutenberg books
    gutenberg_ids = [
        1342, 11, 84, 2701, 98, 74, 76, 158, 160, 161, 163, 166, 172, 174,
        2591, 345, 768, 1232, 1322, 1400, 145, 1952, 2554, 2600, 30254,
        3207, 4300, 5200, 6130, 64317, 1080, 120, 1260, 135, 1399, 1400,
        145, 1497, 1513, 1524, 1533, 158, 160, 161, 163, 164, 166, 172,
        174, 1952, 2554, 2600, 30254, 3207, 4300, 5200, 6130, 64317
    ]
    
    output_path = os.path.join(output_dir, "gutenberg.txt")
    base_url = "https://www.gutenberg.org/files"
    
    collected = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for book_id in tqdm(gutenberg_ids[:num_books], desc="Downloading books"):
            try:
                # Try different file formats
                for suffix in ['-0.txt', '-8.txt', '.txt']:
                    url = f"{base_url}/{book_id}/{book_id}{suffix}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        text = response.text
                        # Clean up Gutenberg headers/footers
                        lines = text.split('\n')
                        start_idx = 0
                        end_idx = len(lines)
                        
                        # Find actual content
                        for i, line in enumerate(lines):
                            if '*** START' in line or '***START' in line:
                                start_idx = i + 1
                            if '*** END' in line or '***END' in line:
                                end_idx = i
                                break
                        
                        content = '\n'.join(lines[start_idx:end_idx])
                        if len(content) > 1000:  # Only save substantial content
                            f.write(content + '\n\n')
                            collected += 1
                        break
            except Exception as e:
                continue
    
    if collected > 0:
        print(f"✓ Collected {collected} books from Project Gutenberg")
        return output_path
    else:
        print("✗ Project Gutenberg download failed")
        return None


def collect_chart_specific_data(output_dir: str):
    """
    Collect chart-specific text data.
    """
    print("Collecting chart-specific data...")
    
    # Create synthetic chart descriptions and analysis
    chart_examples = [
        "This bar chart shows sales data over the past year. The x-axis represents months from January to December, while the y-axis shows sales in thousands of dollars. We can see a steady increase from January through June, with a peak in July, followed by a decline in the fall months.",
        "The line chart displays temperature trends over time. The data shows a clear seasonal pattern with higher temperatures in summer months and lower temperatures in winter. There's a slight upward trend indicating gradual warming over the years.",
        "A pie chart illustrates the market share of different companies. Company A holds the largest share at 35%, followed by Company B at 28%, Company C at 20%, and others making up the remaining 17%.",
        "The scatter plot reveals a strong positive correlation between advertising spend and revenue. As advertising investment increases, revenue shows a corresponding increase, suggesting an effective marketing strategy.",
        "This stacked bar chart compares quarterly performance across different regions. Each bar represents a quarter, with segments showing contributions from North America, Europe, Asia, and other regions.",
    ]
    
    output_path = os.path.join(output_dir, "chart_descriptions.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write examples multiple times with variations
        for i in range(1000):
            for example in chart_examples:
                f.write(example + '\n\n')
    
    print(f"✓ Created chart-specific data: {output_path}")
    return output_path


def combine_text_files(input_files: list, output_file: str):
    """Combine multiple text files into one."""
    print(f"Combining {len(input_files)} files into {output_file}...")
    
    total_size = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(input_files, desc="Combining files"):
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        if len(content) > 100:  # Only add substantial content
                            outfile.write(content + '\n\n')
                            total_size += len(content)
                except Exception as e:
                    print(f"  Warning: Could not read {file_path}: {e}")
    
    size_mb = total_size / (1024 * 1024)
    print(f"✓ Combined {size_mb:.2f} MB of text data")
    return output_file


def collect_pretraining_data(output_dir: str = "./data/pretraining", target_size_gb: float = 5.0):
    """
    Main function to collect pretraining data from multiple sources.
    Target: 5-10 GB for initial training.
    """
    print("=" * 60)
    print("Collecting Pretraining Data")
    print(f"Target: {target_size_gb} GB")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    collected_files = []
    
    # 1. Wikipedia
    wiki_file = download_wikipedia_subset(output_dir)
    if wiki_file:
        collected_files.append(wiki_file)
    
    # 2. Project Gutenberg
    gutenberg_file = download_project_gutenberg(output_dir, num_books=50)
    if gutenberg_file:
        collected_files.append(gutenberg_file)
    
    # 3. Chart-specific data
    chart_file = collect_chart_specific_data(output_dir)
    if chart_file:
        collected_files.append(chart_file)
    
    # 4. OpenWebText (optional, requires datasets library)
    try:
        webtext_file = download_openwebtext_subset(output_dir)
        if webtext_file:
            collected_files.append(webtext_file)
    except:
        print("  Skipping OpenWebText (install datasets library for this)")
    
    # Combine all files
    if collected_files:
        combined_file = os.path.join(output_dir, "combined_pretraining_data.txt")
        combine_text_files(collected_files, combined_file)
        
        # Check size
        size_bytes = os.path.getsize(combined_file)
        size_gb = size_bytes / (1024 ** 3)
        
        print("\n" + "=" * 60)
        print(f"Data Collection Complete!")
        print(f"Total size: {size_gb:.2f} GB")
        print(f"Output: {combined_file}")
        print("=" * 60)
        
        return combined_file
    else:
        print("\n✗ No data collected. Check your internet connection and try again.")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect pretraining data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pretraining",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--target_size_gb",
        type=float,
        default=5.0,
        help="Target data size in GB"
    )
    
    args = parser.parse_args()
    collect_pretraining_data(args.output_dir, args.target_size_gb)


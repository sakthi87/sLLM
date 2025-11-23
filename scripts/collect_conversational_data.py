"""
Collect conversational/instruction data for fine-tuning.
Downloads and prepares 50k-100k instruction-response pairs.
"""
import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_alpaca_dataset(output_dir: str):
    """
    Download Alpaca instruction dataset.
    """
    print("Downloading Alpaca dataset...")
    
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        output_path = os.path.join(output_dir, "alpaca.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Downloaded {len(data)} Alpaca examples")
        return output_path, data
    except Exception as e:
        print(f"✗ Alpaca download failed: {e}")
        return None, []


def download_dolly_dataset(output_dir: str):
    """
    Download Dolly instruction dataset from HuggingFace.
    """
    print("Downloading Dolly dataset...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        output_path = os.path.join(output_dir, "dolly.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to Alpaca format
        data = []
        for example in dataset:
            data.append({
                "instruction": example["instruction"],
                "input": example.get("context", ""),
                "output": example["response"]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Downloaded {len(data)} Dolly examples")
        return output_path, data
    except Exception as e:
        print(f"✗ Dolly download failed: {e}")
        print("  Install: pip install datasets")
        return None, []


def download_sharegpt_subset(output_dir: str, max_examples: int = 10000):
    """
    Download ShareGPT conversation dataset.
    """
    print("Downloading ShareGPT dataset...")
    
    try:
        from datasets import load_dataset
        
        # ShareGPT is large, so we'll stream and limit
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        
        output_path = os.path.join(output_dir, "sharegpt.json")
        os.makedirs(output_dir, exist_ok=True)
        
        data = []
        count = 0
        
        for example in tqdm(dataset, desc="Collecting ShareGPT", total=max_examples):
            if count >= max_examples:
                break
            
            conversations = example.get("conversations", [])
            if conversations and len(conversations) > 0:
                # Convert to instruction format
                messages = []
                for conv in conversations:
                    role = conv.get("from", "").lower()
                    content = conv.get("value", "")
                    if role and content:
                        messages.append({"role": role, "content": content})
                
                if len(messages) >= 2:  # At least user and assistant
                    data.append({"messages": messages})
                    count += 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Collected {len(data)} ShareGPT conversations")
        return output_path, data
    except Exception as e:
        print(f"✗ ShareGPT download failed: {e}")
        print("  Install: pip install datasets")
        return None, []


def create_synthetic_chart_conversations(output_dir: str, num_examples: int = 5000):
    """
    Create synthetic chart-related conversations.
    """
    print(f"Creating {num_examples} synthetic chart conversations...")
    
    chart_topics = [
        ("What does this bar chart show?", "This bar chart displays sales data across different months, with the height of each bar representing the sales value for that period."),
        ("Explain the trend in this line graph.", "The line graph shows a steady upward trend over time, indicating consistent growth in the metric being measured."),
        ("What insights can you draw from this pie chart?", "The pie chart reveals that Category A represents the largest portion at 40%, followed by Category B at 30%, with the remaining segments making up 30%."),
        ("How would you interpret this scatter plot?", "The scatter plot shows a positive correlation between the two variables, suggesting that as one increases, the other tends to increase as well."),
        ("What does this heatmap indicate?", "The heatmap visualizes the relationship between different categories, with darker colors representing higher values and lighter colors representing lower values."),
    ]
    
    log_topics = [
        ("What does this Cassandra error mean?", "This Cassandra error indicates a connection timeout, which typically occurs when the database cannot establish a connection within the specified time limit."),
        ("Explain this Spark log entry.", "This Spark log shows that a job completed successfully, processing the data in the expected timeframe."),
        ("What should I do about this warning?", "This warning suggests monitoring the system closely, as it may indicate a potential issue that could escalate if not addressed."),
    ]
    
    metadata_topics = [
        ("What is the schema of this table?", "The table has columns for id, name, email, and created_at, with id serving as the primary key."),
        ("How does this API work?", "This API accepts a GET request with an ID parameter and returns the corresponding resource data in JSON format."),
        ("What does this Spark job do?", "This Spark job processes daily data, aggregating metrics and writing the results to an output table."),
    ]
    
    all_topics = chart_topics + log_topics + metadata_topics
    
    data = []
    for i in range(num_examples):
        topic_idx = i % len(all_topics)
        question, answer = all_topics[topic_idx]
        
        # Add variations
        variations = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # Sometimes add follow-up
        if i % 3 == 0 and len(all_topics) > topic_idx + 1:
            next_q, next_a = all_topics[(topic_idx + 1) % len(all_topics)]
            variations.append({"role": "user", "content": f"Can you tell me more about {next_q.lower()}"})
            variations.append({"role": "assistant", "content": next_a})
        
        data.append({"messages": variations})
    
    output_path = os.path.join(output_dir, "synthetic_chart_conversations.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {len(data)} synthetic conversations")
    return output_path, data


def convert_to_chat_format(data_list: list, source_name: str):
    """
    Convert various formats to unified chat format.
    """
    converted = []
    
    for item in data_list:
        if "messages" in item:
            # Already in chat format
            converted.append(item)
        elif "instruction" in item:
            # Alpaca/Dolly format
            messages = [
                {"role": "user", "content": item["instruction"] + ("\n" + item.get("input", "") if item.get("input") else "")},
                {"role": "assistant", "content": item["output"]}
            ]
            converted.append({"messages": messages})
        elif "conversations" in item:
            # ShareGPT format
            messages = []
            for conv in item["conversations"]:
                role = conv.get("from", "").lower()
                if role == "human":
                    role = "user"
                elif role == "gpt" or role == "assistant":
                    role = "assistant"
                messages.append({"role": role, "content": conv.get("value", "")})
            if messages:
                converted.append({"messages": messages})
    
    return converted


def combine_conversational_data(output_dir: str):
    """
    Combine all conversational datasets into one file.
    """
    print("\nCombining all conversational datasets...")
    
    all_data = []
    
    # Load all JSON files in output_dir
    for file_path in Path(output_dir).glob("*.json"):
        if file_path.name == "combined_conversations.json":
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    converted = convert_to_chat_format(data, file_path.stem)
                    all_data.extend(converted)
                    print(f"  Added {len(converted)} from {file_path.name}")
        except Exception as e:
            print(f"  Warning: Could not load {file_path}: {e}")
    
    # Save combined
    output_path = os.path.join(output_dir, "combined_conversations.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined {len(all_data)} total conversations")
    print(f"  Saved to: {output_path}")
    
    return output_path


def collect_conversational_data(output_dir: str = "./data/conversational", target_examples: int = 50000):
    """
    Main function to collect conversational/instruction data.
    Target: 50k-100k examples.
    """
    print("=" * 60)
    print("Collecting Conversational/Instruction Data")
    print(f"Target: {target_examples:,} examples")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Alpaca
    alpaca_path, alpaca_data = download_alpaca_dataset(output_dir)
    
    # 2. Dolly
    dolly_path, dolly_data = download_dolly_dataset(output_dir)
    
    # 3. ShareGPT (subset)
    sharegpt_path, sharegpt_data = download_sharegpt_subset(output_dir, max_examples=10000)
    
    # 4. Synthetic chart conversations
    synthetic_path, synthetic_data = create_synthetic_chart_conversations(output_dir, num_examples=5000)
    
    # 5. Combine all
    combined_path = combine_conversational_data(output_dir)
    
    # Summary
    total = len(alpaca_data) + len(dolly_data) + len(sharegpt_data) + len(synthetic_data)
    
    print("\n" + "=" * 60)
    print("Data Collection Summary:")
    print(f"  Alpaca: {len(alpaca_data):,} examples")
    print(f"  Dolly: {len(dolly_data):,} examples")
    print(f"  ShareGPT: {len(sharegpt_data):,} examples")
    print(f"  Synthetic: {len(synthetic_data):,} examples")
    print(f"  Total: {total:,} examples")
    print("=" * 60)
    
    return combined_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect conversational data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/conversational",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--target_examples",
        type=int,
        default=50000,
        help="Target number of examples"
    )
    
    args = parser.parse_args()
    collect_conversational_data(args.output_dir, args.target_examples)


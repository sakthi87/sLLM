"""
Train a custom tokenizer from scratch on your data.
This creates a BPE (Byte-Pair Encoding) tokenizer optimized for your corpus.
"""
import os
import sys
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import ByteLevel
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_tokenizer(
    data_files: list,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2
):
    """
    Train a BPE tokenizer on the provided data files.
    
    Args:
        data_files: List of file paths containing training text
        output_dir: Directory to save the tokenizer
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
    """
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    print(f"Data files: {data_files}")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = ByteLevel(trim_offsets=True)
    
    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<sep>"]
    )
    
    # Train tokenizer
    print("Training on data files...")
    # Convert all paths to strings
    data_files_str = [str(f) if not isinstance(f, str) else f for f in data_files]
    tokenizer.train(files=data_files_str, trainer=trainer)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Also save as HuggingFace format for compatibility
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        sep_token="<sep>"
    )
    hf_tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace tokenizer saved to {output_dir}")
    
    # Print some statistics
    print(f"\nTokenizer Statistics:")
    print(f"  Vocabulary size: {len(tokenizer.get_vocab())}")
    print(f"  Special tokens: {trainer.special_tokens}")
    
    return tokenizer


def prepare_text_files(data_dir: str, output_file: str = None):
    """
    Prepare text files from various data sources.
    Extracts text from JSON files and creates plain text files.
    """
    text_files = []
    
    # Process chat data
    chat_dir = os.path.join(data_dir, "chat")
    if os.path.exists(chat_dir):
        for file in Path(chat_dir).glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                # Extract text from messages
                texts = []
                for item in data:
                    if "messages" in item:
                        for msg in item["messages"]:
                            texts.append(msg.get("content", ""))
                    elif "text" in item:
                        texts.append(item["text"])
                
                # Write to temporary text file
                temp_file = str(file).replace(".json", "_text.txt")
                with open(temp_file, 'w') as out:
                    out.write("\n".join(texts))
                text_files.append(temp_file)
    
    # Process log data
    logs_dir = os.path.join(data_dir, "logs")
    if os.path.exists(logs_dir):
        for file in Path(logs_dir).glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                texts = []
                for item in data:
                    texts.append(item.get("log_entry", ""))
                    texts.append(item.get("question", ""))
                    texts.append(item.get("answer", ""))
                
                temp_file = str(file).replace(".json", "_text.txt")
                with open(temp_file, 'w') as out:
                    out.write("\n".join(texts))
                text_files.append(temp_file)
    
    # Process metadata
    metadata_dir = os.path.join(data_dir, "metadata")
    if os.path.exists(metadata_dir):
        for file in Path(metadata_dir).glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                # Convert JSON to text representation
                text = json.dumps(data, indent=2)
                temp_file = str(file).replace(".json", "_text.txt")
                with open(temp_file, 'w') as out:
                    out.write(text)
                text_files.append(temp_file)
    
    # Also look for plain text files
    for ext in ["*.txt", "*.text"]:
        text_files.extend(list(Path(data_dir).rglob(ext)))
    
    return text_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom tokenizer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=None,
        help="Specific data files to use (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer",
        help="Directory to save the trained tokenizer"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens"
    )
    
    args = parser.parse_args()
    
    # Get data files
    if args.data_files:
        data_files = args.data_files
    else:
        print("Auto-detecting data files...")
        data_files = prepare_text_files(args.data_dir)
        if not data_files:
            print("Warning: No data files found. Please provide --data_files or ensure data exists.")
            sys.exit(1)
    
    print(f"Found {len(data_files)} data files")
    
    # Train tokenizer
    train_tokenizer(
        data_files=data_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )
    
    print("\nTokenizer training complete!")


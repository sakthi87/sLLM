"""
Inference script for chatting with the trained from-scratch model.
"""
import os
import sys
import torch
from transformers import PreTrainedTokenizerFast
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import ChartLanguageModel


def load_model_for_inference(model_path: str, tokenizer_path: str = None):
    """Load model and tokenizer for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    if tokenizer_path is None:
        # Try to find tokenizer in model directory
        tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.dirname(model_path)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Load model checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]["model"]
        vocab_size = checkpoint.get("vocab_size", len(tokenizer))
    else:
        # Default config
        config = {
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "d_ff": 3072,
            "max_seq_len": 512,
            "dropout": 0.1
        }
        vocab_size = len(tokenizer)
    
    # Initialize model
    model = ChartLanguageModel(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        d_ff=config.get("d_ff", 3072),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    num_params = model.get_num_params()
    print(f"Model loaded with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    return model, tokenizer, device


def generate_response(
    model, 
    tokenizer, 
    device,
    prompt: str, 
    max_new_tokens: int = 100, 
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9
):
    """Generate a response from the model."""
    # Format prompt
    formatted_prompt = f"User: {prompt}\n\nAssistant:"
    
    # Tokenize
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    input_ids = encoded["input_ids"].to(device)
    
    # Generate using model's generate method
    # Use better parameters to avoid repetition
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 0.8),  # Ensure minimum temperature
        top_k=min(top_k, 40) if top_k else 40,  # Limit top_k
        top_p=min(top_p, 0.95),  # Limit top_p
        do_sample=True,
        repetition_penalty=1.3  # Add repetition penalty
    )
    
    # Decode
    full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        # If format not found, return the generated part
        prompt_tokens = len(input_ids[0])
        response = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)
    
    return response


def interactive_chat(model_path: str, tokenizer_path: str = None):
    """Interactive chat interface."""
    print("Loading model...")
    model, tokenizer, device = load_model_for_inference(model_path, tokenizer_path)
    print("Model loaded! You can start chatting. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        print("Assistant: ", end="", flush=True)
        try:
            response = generate_response(model, tokenizer, device, user_input)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the trained from-scratch model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: looks in model directory)"
    )
    
    args = parser.parse_args()
    
    interactive_chat(args.model_path, args.tokenizer_path)

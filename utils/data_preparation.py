"""
Data preparation utilities for building language model from scratch.
Simplified for from-scratch training without HuggingFace dependencies.
"""
import json
import os
from typing import List, Dict, Any


def prepare_chat_format(messages: List[Dict[str, str]]) -> str:
    """
    Convert chat messages to a formatted string for training.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
        
    Returns:
        Formatted conversation string
    """
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    
    return formatted.strip()


def load_text_from_chat_data(json_path: str) -> List[str]:
    """
    Load chat data from JSON and extract text for training.
    
    Expected JSON format:
    [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ]
        },
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        if "messages" in item:
            # Format the conversation
            formatted = prepare_chat_format(item["messages"])
            texts.append(formatted)
        elif "text" in item:
            texts.append(item["text"])
    
    return texts


def load_text_from_logs(json_path: str) -> List[str]:
    """
    Load log data and create text examples.
    
    Expected format:
    [
        {
            "log_entry": "...",
            "question": "...",
            "answer": "..."
        }
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    texts = []
    for log in logs:
        log_entry = log.get("log_entry", "")
        question = log.get("question", "")
        answer = log.get("answer", "")
        
        if log_entry and question and answer:
            text = f"Log: {log_entry}\nQuestion: {question}\nAnswer: {answer}"
            texts.append(text)
    
    return texts


def load_text_from_metadata(json_path: str) -> List[str]:
    """
    Load metadata and convert to text format.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Convert to text representation
    text = json.dumps(metadata, indent=2)
    return [text]


def collect_all_texts(data_dir: str) -> List[str]:
    """
    Collect all text data from various sources in the data directory.
    
    Returns:
        List of text strings for training
    """
    all_texts = []
    
    # Load chat data
    chat_dir = os.path.join(data_dir, "chat")
    if os.path.exists(chat_dir):
        for file in os.listdir(chat_dir):
            if file.endswith(".json"):
                file_path = os.path.join(chat_dir, file)
                texts = load_text_from_chat_data(file_path)
                all_texts.extend(texts)
                print(f"Loaded {len(texts)} texts from {file}")
    
    # Load log data
    logs_dir = os.path.join(data_dir, "logs")
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if file.endswith(".json"):
                file_path = os.path.join(logs_dir, file)
                texts = load_text_from_logs(file_path)
                all_texts.extend(texts)
                print(f"Loaded {len(texts)} texts from {file}")
    
    # Load metadata
    metadata_dir = os.path.join(data_dir, "metadata")
    if os.path.exists(metadata_dir):
        for file in os.listdir(metadata_dir):
            if file.endswith(".json"):
                file_path = os.path.join(metadata_dir, file)
                texts = load_text_from_metadata(file_path)
                all_texts.extend(texts)
                print(f"Loaded {len(texts)} texts from {file}")
    
    # Load plain text files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".txt", ".text")):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split long files into chunks
                    if len(content) > 10000:
                        chunks = [content[i:i+10000] for i in range(0, len(content), 10000)]
                        all_texts.extend(chunks)
                    else:
                        all_texts.append(content)
                print(f"Loaded text from {file}")
    
    return all_texts

"""
From-scratch GPT-style Transformer model for conversational language modeling.
This is a clean implementation without relying on pre-trained models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More efficient and stable than LayerNorm, especially for FP16 training.
    Used by modern models like LLaMA, TinyLlama, Phi, Falcon.
    
    Reference: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (no bias term)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model] or [batch_size, d_model]
        Returns:
            Normalized tensor with same shape
        """
        # Calculate RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_rmsnorm: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # Use RMSNorm for better performance and FP16 stability
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ChartLanguageModel(nn.Module):
    """
    GPT-style language model built from scratch.
    Designed for conversational chart-related language understanding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_rmsnorm: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks (use RMSNorm by default)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rmsnorm=use_rmsnorm)
            for _ in range(n_layers)
        ])
        
        # Final layer norm (use RMSNorm by default)
        if use_rmsnorm:
            self.ln_f = RMSNorm(d_model)
        else:
            self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights: output embedding = input embedding (common in GPT models)
        self.head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Optional scalar loss if labels provided
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create causal mask (prevent attending to future tokens)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
        # [1, seq_len, seq_len]
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand attention mask to [batch_size, 1, seq_len, seq_len]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * padding_mask
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
                label_smoothing=label_smoothing
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalty)
        
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for last token
                logits, _ = self.forward(generated)
                logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Get recent tokens (last 20 tokens)
                    recent_tokens = generated[:, -20:] if generated.size(1) > 20 else generated
                    for token_id in recent_tokens[0].unique():
                        logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    # Avoid sampling tokens with very low probability
                    probs = torch.clamp(probs, min=1e-10)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Stop if we hit padding token or end token
                if next_token.item() == self.pad_token_id:
                    break
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


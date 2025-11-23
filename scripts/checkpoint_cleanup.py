#!/usr/bin/env python3
"""
Checkpoint cleanup utility - removes old checkpoints while keeping:
- Last N checkpoints
- Milestone checkpoints (every 10th epoch)
- Best checkpoint (lowest loss)
"""

import os
import glob
import torch
from pathlib import Path


def cleanup_checkpoints(
    output_dir: str,
    keep_last: int = 5,
    keep_milestones: bool = True,
    milestone_interval: int = 10,
    keep_best: bool = True
):
    """
    Clean up old checkpoints, keeping only the most important ones.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last: Number of most recent checkpoints to keep
        keep_milestones: Whether to keep milestone checkpoints (every Nth epoch)
        milestone_interval: Interval for milestone checkpoints (e.g., 10 = every 10th epoch)
        keep_best: Whether to keep the checkpoint with lowest loss
    """
    # Find all checkpoints
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_epoch_*.pt"))
    
    if len(checkpoint_files) <= keep_last:
        return 0  # Not enough checkpoints to clean
    
    # Extract epoch numbers and load losses
    checkpoints = []
    for cp_file in checkpoint_files:
        try:
            # Extract epoch number from filename
            epoch_num = int(Path(cp_file).stem.split("_")[-1])
            
            # Load checkpoint to get loss
            try:
                ckpt = torch.load(cp_file, map_location='cpu')
                # Try to get loss (could be 'loss', 'train_loss', or 'val_loss')
                loss = ckpt.get('loss', ckpt.get('train_loss', ckpt.get('val_loss', float('inf'))))
            except:
                loss = float('inf')
            
            checkpoints.append({
                'file': cp_file,
                'epoch': epoch_num,
                'loss': loss
            })
        except:
            continue
    
    if not checkpoints:
        return 0
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    
    # Determine which checkpoints to keep
    keep_files = set()
    
    # 1. Keep last N checkpoints
    for cp in checkpoints[-keep_last:]:
        keep_files.add(cp['file'])
    
    # 2. Keep milestone checkpoints
    if keep_milestones:
        for cp in checkpoints:
            if cp['epoch'] % milestone_interval == 0:
                keep_files.add(cp['file'])
    
    # 3. Keep best checkpoint (lowest loss)
    if keep_best:
        best_cp = min(checkpoints, key=lambda x: x['loss'])
        keep_files.add(best_cp['file'])
    
    # Delete checkpoints not in keep_files
    deleted_count = 0
    deleted_size = 0
    for cp in checkpoints:
        if cp['file'] not in keep_files:
            try:
                size = os.path.getsize(cp['file'])
                os.remove(cp['file'])
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                print(f"⚠️  Failed to delete {cp['file']}: {e}")
    
    return deleted_count, deleted_size


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old checkpoints")
    parser.add_argument("--output-dir", type=str, default="./models/pretrained",
                       help="Directory containing checkpoints")
    parser.add_argument("--keep-last", type=int, default=5,
                       help="Number of most recent checkpoints to keep")
    parser.add_argument("--keep-milestones", action="store_true", default=True,
                       help="Keep milestone checkpoints (every 10th epoch)")
    parser.add_argument("--milestone-interval", type=int, default=10,
                       help="Interval for milestone checkpoints")
    parser.add_argument("--keep-best", action="store_true", default=True,
                       help="Keep checkpoint with lowest loss")
    
    args = parser.parse_args()
    
    deleted_count, deleted_size = cleanup_checkpoints(
        args.output_dir,
        keep_last=args.keep_last,
        keep_milestones=args.keep_milestones,
        milestone_interval=args.milestone_interval,
        keep_best=args.keep_best
    )
    
    print(f"✅ Cleaned up {deleted_count} checkpoints")
    print(f"   Freed {deleted_size / (1024**3):.2f} GB")


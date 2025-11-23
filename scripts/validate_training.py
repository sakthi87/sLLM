#!/usr/bin/env python3
"""
Validate training progress and health.
Can be run anytime during training to check status.
"""
import os
import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime

def check_process_running():
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'pretrain.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    return {
                        'running': True,
                        'pid': pid,
                        'cpu': cpu,
                        'mem': mem
                    }
        return {'running': False}
    except:
        return {'running': False, 'error': 'Could not check process'}

def check_loss_trend(log_file='/tmp/pretraining_1024_optimized.log'):
    """Extract and analyze loss values from log."""
    if not os.path.exists(log_file):
        return {'error': f'Log file not found: {log_file}'}
    
    losses = []
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Extract loss values
            pattern = r'loss=([0-9]+\.[0-9]+)'
            matches = re.findall(pattern, content)
            losses = [float(m) for m in matches]
    except Exception as e:
        return {'error': f'Could not read log: {e}'}
    
    if len(losses) < 2:
        return {'error': 'Not enough loss data yet'}
    
    first_loss = losses[0]
    recent_losses = losses[-20:] if len(losses) > 20 else losses
    latest_loss = losses[-1]
    avg_recent = sum(recent_losses) / len(recent_losses)
    
    # Determine trend
    if latest_loss < first_loss:
        trend = 'decreasing'
        status = '✅'
    elif latest_loss > first_loss * 1.1:  # 10% increase
        trend = 'increasing'
        status = '⚠️'
    else:
        trend = 'stable'
        status = '✅'
    
    return {
        'first_loss': first_loss,
        'latest_loss': latest_loss,
        'avg_recent': avg_recent,
        'trend': trend,
        'status': status,
        'total_samples': len(losses)
    }

def check_progress(log_file='/tmp/pretraining_1024_optimized.log'):
    """Check current training progress."""
    if not os.path.exists(log_file):
        return {'error': f'Log file not found: {log_file}'}
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Get last line with progress
            for line in reversed(lines):
                if 'Training:' in line and '/' in line:
                    # Extract step info: "Training:  51%|█████     | 3123/6167"
                    match = re.search(r'(\d+)/(\d+)', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        percent = (current / total) * 100
                        
                        # Extract epoch
                        epoch_match = re.search(r'Epoch (\d+)/(\d+)', '\n'.join(lines[-100:]))
                        epoch = None
                        total_epochs = None
                        if epoch_match:
                            epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                        
                        return {
                            'current_step': current,
                            'total_steps': total,
                            'percent': percent,
                            'epoch': epoch,
                            'total_epochs': total_epochs
                        }
        return {'error': 'Could not find progress info'}
    except Exception as e:
        return {'error': f'Error reading log: {e}'}

def check_checkpoints(checkpoint_dir='models/pretrained'):
    """Check if checkpoints are being created."""
    if not os.path.exists(checkpoint_dir):
        return {'error': f'Checkpoint directory not found: {checkpoint_dir}'}
    
    checkpoints = []
    for file in Path(checkpoint_dir).glob('checkpoint_epoch_*.pt'):
        if 'error' not in file.name and 'backup' not in file.name:
            try:
                # Extract epoch number
                epoch = int(re.search(r'epoch_(\d+)', file.name).group(1))
                size = file.stat().st_size / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                checkpoints.append({
                    'epoch': epoch,
                    'size_mb': size,
                    'modified': mtime
                })
            except:
                pass
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x['epoch'])
        return {
            'found': True,
            'count': len(checkpoints),
            'latest_epoch': checkpoints[-1]['epoch'],
            'latest_size_mb': checkpoints[-1]['size_mb'],
            'latest_modified': checkpoints[-1]['modified']
        }
    else:
        return {'found': False, 'count': 0}

def check_memory_usage():
    """Check system memory usage."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'pretrain.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 5:
                    # Memory in KB, convert to GB
                    mem_kb = float(parts[5])
                    mem_gb = mem_kb / (1024 * 1024)
                    return {
                        'memory_gb': mem_gb,
                        'status': '✅' if mem_gb < 6 else '⚠️' if mem_gb < 8 else '❌'
                    }
        return {'error': 'Process not found'}
    except Exception as e:
        return {'error': f'Could not check memory: {e}'}

def main():
    print("=" * 70)
    print("🔍 TRAINING VALIDATION REPORT")
    print("=" * 70)
    print()
    
    # 1. Process Status
    print("1️⃣  PROCESS STATUS")
    print("-" * 70)
    proc = check_process_running()
    if proc.get('running'):
        print(f"   ✅ Training is RUNNING")
        print(f"   PID: {proc['pid']}")
        print(f"   CPU: {proc['cpu']}%")
        print(f"   Memory: {proc['mem']}%")
    else:
        print(f"   ❌ Training is NOT running")
        if 'error' in proc:
            print(f"   Error: {proc['error']}")
    print()
    
    # 2. Progress
    print("2️⃣  TRAINING PROGRESS")
    print("-" * 70)
    progress = check_progress()
    if 'error' not in progress:
        print(f"   Epoch: {progress.get('epoch', '?')}/{progress.get('total_epochs', '?')}")
        print(f"   Steps: {progress.get('current_step', '?')}/{progress.get('total_steps', '?')}")
        print(f"   Progress: {progress.get('percent', 0):.1f}%")
    else:
        print(f"   ⚠️  {progress['error']}")
    print()
    
    # 3. Loss Trend
    print("3️⃣  LOSS TREND")
    print("-" * 70)
    loss_info = check_loss_trend()
    if 'error' not in loss_info:
        print(f"   {loss_info['status']} First loss: {loss_info['first_loss']:.4f}")
        print(f"   {loss_info['status']} Latest loss: {loss_info['latest_loss']:.4f}")
        print(f"   {loss_info['status']} Recent avg: {loss_info['avg_recent']:.4f}")
        print(f"   {loss_info['status']} Trend: {loss_info['trend'].upper()}")
        print(f"   Samples analyzed: {loss_info['total_samples']}")
        
        if loss_info['trend'] == 'decreasing':
            print(f"   ✅ Loss is decreasing - training is working!")
        elif loss_info['trend'] == 'increasing':
            print(f"   ⚠️  Loss is increasing - may need investigation")
        else:
            print(f"   ✅ Loss is stable - normal during training")
    else:
        print(f"   ⚠️  {loss_info['error']}")
    print()
    
    # 4. Checkpoints
    print("4️⃣  CHECKPOINTS")
    print("-" * 70)
    ckpt = check_checkpoints()
    if 'error' not in ckpt:
        if ckpt.get('found'):
            print(f"   ✅ Found {ckpt['count']} checkpoint(s)")
            print(f"   Latest: Epoch {ckpt['latest_epoch']}")
            print(f"   Size: {ckpt['latest_size_mb']:.1f} MB")
            print(f"   Modified: {ckpt['latest_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   ⚠️  No checkpoints found yet (may be too early)")
    else:
        print(f"   ⚠️  {ckpt['error']}")
    print()
    
    # 5. Memory Usage
    print("5️⃣  MEMORY USAGE")
    print("-" * 70)
    mem = check_memory_usage()
    if 'error' not in mem:
        print(f"   {mem['status']} Memory: {mem['memory_gb']:.2f} GB")
        if mem['memory_gb'] < 6:
            print(f"   ✅ Memory usage is good")
        elif mem['memory_gb'] < 8:
            print(f"   ⚠️  Memory usage is moderate")
        else:
            print(f"   ❌ Memory usage is high - may cause swapping")
    else:
        print(f"   ⚠️  {mem['error']}")
    print()
    
    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    all_good = True
    if not proc.get('running'):
        print("   ❌ Training process is not running")
        all_good = False
    if 'error' in loss_info or (loss_info.get('trend') == 'increasing'):
        print("   ⚠️  Loss trend may need attention")
    if not ckpt.get('found'):
        print("   ⚠️  No checkpoints yet (may be normal if just started)")
    if 'error' in mem or (mem.get('memory_gb', 0) > 8):
        print("   ⚠️  Memory usage may be high")
    
    if all_good and proc.get('running'):
        print("   ✅ Training appears to be running normally!")
        print("   ✅ Continue monitoring - training is progressing")
    
    print()
    print("💡 Run this script anytime to check training status")
    print("=" * 70)

if __name__ == "__main__":
    main()


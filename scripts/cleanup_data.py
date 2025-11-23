#!/usr/bin/env python3
"""
Clean up data directory before collecting 50GB.
Removes empty files, broken files, and organizes structure.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup_data_directory(data_dir: str = "./data", dry_run: bool = True):
    """
    Clean up data directory.
    
    Args:
        data_dir: Path to data directory
        dry_run: If True, only show what would be removed (don't actually remove)
    """
    print("="*70)
    print("DATA DIRECTORY CLEANUP")
    print("="*70)
    print()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    empty_files = []
    broken_files = []
    total_size_freed = 0
    
    # Scan all files
    print("🔍 Scanning for cleanup...")
    print()
    
    for file in data_path.rglob("*"):
        if file.is_file():
            try:
                size = file.stat().st_size
                if size == 0:
                    empty_files.append(file)
                elif size < 100:  # Very small files (likely broken)
                    broken_files.append(file)
            except Exception as e:
                broken_files.append(file)
                print(f"  ⚠️  Error checking {file}: {e}")
    
    # Report findings
    print(f"📊 Cleanup Analysis:")
    print(f"   Empty files: {len(empty_files)}")
    print(f"   Broken/small files: {len(broken_files)}")
    print()
    
    if empty_files:
        print("Empty files found:")
        for f in empty_files:
            print(f"  • {f.relative_to(data_path)}")
        print()
    
    if broken_files:
        print("Broken/very small files found:")
        for f in broken_files[:10]:  # Show first 10
            try:
                size = f.stat().st_size
                print(f"  • {f.relative_to(data_path)} ({size} bytes)")
            except:
                print(f"  • {f.relative_to(data_path)} (error)")
        if len(broken_files) > 10:
            print(f"  ... and {len(broken_files) - 10} more")
        print()
    
    # Remove files if not dry run
    if not dry_run:
        print("🗑️  Removing files...")
        print()
        
        removed = 0
        for file in empty_files + broken_files:
            try:
                file.unlink()
                removed += 1
                print(f"  ✅ Removed: {file.relative_to(data_path)}")
            except Exception as e:
                print(f"  ❌ Failed to remove {file.relative_to(data_path)}: {e}")
        
        print()
        print(f"✅ Cleanup complete! Removed {removed} files")
    else:
        print("ℹ️  DRY RUN MODE - No files were removed")
        print()
        print("To actually remove files, run with --execute flag:")
        print("  python scripts/cleanup_data.py --execute")
    
    print()
    print("="*70)
    print("DATA DIRECTORY STRUCTURE (After Cleanup):")
    print("="*70)
    print()
    
    # Show directory structure
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            files = list(subdir.rglob("*"))
            files = [f for f in files if f.is_file()]
            total_size = sum(f.stat().st_size for f in files if f.exists()) / (1024*1024)
            
            print(f"  {subdir.name}/")
            print(f"    Files: {len(files)}")
            print(f"    Size: {total_size:.2f} MB")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up data directory")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove files (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    cleanup_data_directory(
        data_dir=args.data_dir,
        dry_run=not args.execute
    )


#!/usr/bin/env python3
"""
Script to verify dataset structure for YOWOFormer
"""

import os
from pathlib import Path
import json


def check_ucf101_24():
    """Check UCF101-24 dataset structure"""
    print("\n" + "="*50)
    print("Checking UCF101-24 Dataset")
    print("="*50)
    
    base_path = Path("data/UCF101-24")
    
    if not base_path.exists():
        print(f"❌ UCF101-24 directory not found: {base_path.absolute()}")
        return False
    
    print(f"✅ Found UCF101-24 at: {base_path.absolute()}")
    
    # Check required directories
    required_dirs = {
        "rgb-images": "Frame images directory",
        "labels": "Annotation files directory"
    }
    
    for dir_name, desc in required_dirs.items():
        dir_path = base_path / dir_name
        if dir_path.exists():
            # Count subdirectories (videos)
            subdirs = list(dir_path.iterdir())
            print(f"  ✅ {dir_name}/: {desc} - Found {len(subdirs)} items")
        else:
            print(f"  ❌ {dir_name}/: {desc} - NOT FOUND")
    
    # Check required files
    required_files = {
        "trainlist.txt": "Training video list",
        "testlist.txt": "Testing video list"
    }
    
    for file_name, desc in required_files.items():
        file_path = base_path / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            print(f"  ✅ {file_name}: {desc} - {len(lines)} videos")
        else:
            print(f"  ❌ {file_name}: {desc} - NOT FOUND")
    
    # Check sample data
    rgb_path = base_path / "rgb-images"
    if rgb_path.exists():
        videos = list(rgb_path.iterdir())
        if videos:
            sample_video = videos[0]
            frames = list(sample_video.iterdir())
            print(f"\n  Sample video: {sample_video.name}")
            print(f"    Frames: {len(frames)}")
            
            # Check corresponding labels
            label_file = base_path / "labels" / f"{sample_video.name}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                print(f"    Annotations: {len(annotations)} lines")
            else:
                print(f"    ❌ No label file found")
    
    return True


def check_ava_v22():
    """Check AVA v2.2 dataset structure"""
    print("\n" + "="*50)
    print("Checking AVA v2.2 Dataset")
    print("="*50)
    
    base_path = Path("data/AVA_v2.2")
    
    if not base_path.exists():
        print(f"❌ AVA v2.2 directory not found: {base_path.absolute()}")
        return False
    
    print(f"✅ Found AVA v2.2 at: {base_path.absolute()}")
    
    # Check frames directory
    frames_path = base_path / "frames"
    if frames_path.exists():
        videos = list(frames_path.iterdir())
        print(f"  ✅ frames/: Found {len(videos)} videos")
        
        if videos:
            sample_video = videos[0]
            frames = list(sample_video.iterdir())
            print(f"    Sample video: {sample_video.name}")
            print(f"    Frames: {len(frames)}")
    else:
        print(f"  ❌ frames/: Frame directory - NOT FOUND")
    
    # Check annotation files
    annotation_files = {
        "ava_train_v2.2.csv": "Training annotations",
        "ava_val_v2.2.csv": "Validation annotations",
        "ava_action_list_v2.2.pbtxt": "Action class list",
        "ava_train_excluded_timestamps_v2.2.csv": "Excluded timestamps (train)",
        "ava_val_excluded_timestamps_v2.2.csv": "Excluded timestamps (val)"
    }
    
    print("\n  Annotation files:")
    for file_name, desc in annotation_files.items():
        file_path = base_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file_name}: {desc} ({size_mb:.2f} MB)")
        else:
            # Check in annotations subdirectory
            file_path = base_path / "annotations" / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ annotations/{file_name}: {desc} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {file_name}: {desc} - NOT FOUND")
    
    return True


def check_dataset_config():
    """Check dataset configuration files"""
    print("\n" + "="*50)
    print("Checking Dataset Configurations")
    print("="*50)
    
    config_files = [
        "YOWOFormer/config/ucf101-24/ucf101-24.yaml",
        "YOWOFormer/config/ava/ava_v2.2.yaml"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            print(f"  ✅ {config_file}")
            # Could parse YAML and check data paths here
        else:
            print(f"  ❌ {config_file} - NOT FOUND")


def main():
    """Main verification function"""
    print("\n" + "🔍 YOWOFormer Dataset Verification Tool" + "\n")
    
    # Check UCF101-24
    ucf_ok = check_ucf101_24()
    
    # Check AVA v2.2
    ava_ok = check_ava_v22()
    
    # Check configs
    check_dataset_config()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if ucf_ok and ava_ok:
        print("✅ All datasets are ready!")
        print("\nNext steps:")
        print("1. Update config files with correct data paths")
        print("2. Test data loading with sample script")
        print("3. Start training!")
    else:
        print("⚠️ Some datasets are missing or incomplete")
        print("\nPlease:")
        print("1. Download missing datasets")
        print("2. Extract files to correct directories")
        print("3. Run this script again to verify")
    
    print("\n📁 Expected directory structure:")
    print("""
    data/
    ├── UCF101-24/
    │   ├── rgb-images/      # Video frames
    │   ├── labels/          # Annotations
    │   ├── trainlist.txt
    │   └── testlist.txt
    └── AVA_v2.2/
        ├── frames/          # Video frames
        ├── annotations/     # CSV files (optional)
        ├── ava_train_v2.2.csv
        ├── ava_val_v2.2.csv
        └── ava_action_list_v2.2.pbtxt
    """)


if __name__ == "__main__":
    main()
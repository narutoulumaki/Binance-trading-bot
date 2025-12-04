"""
Pascal VOC Dataset Download Script
Downloads VOC2007 and VOC2012 datasets for object detection training.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

# Dataset URLs
VOC_URLS = {
    'VOC2007_trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'VOC2007_test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'VOC2012_trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
}

def download_file(url, dest_path):
    """Download file with progress bar."""
    print(f"\n📥 Downloading {url.split('/')[-1]}...")
    
    def progress_hook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        print(f"\rProgress: {'█' * (percent // 2)}{' ' * (50 - percent // 2)} {percent}%", end='')
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_tar(tar_path, extract_path):
    """Extract tar archive."""
    print(f"\n📦 Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_path)
        print("✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir
    voc_dir = data_dir / 'VOCdevkit'
    
    print("=" * 70)
    print("🎯 Pascal VOC Dataset Downloader")
    print("=" * 70)
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 VOC directory: {voc_dir}")
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    voc_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and extract each dataset
    for name, url in VOC_URLS.items():
        tar_filename = url.split('/')[-1]
        tar_path = data_dir / tar_filename
        
        # Check if already downloaded
        if tar_path.exists():
            print(f"\n⏭️  {tar_filename} already exists, skipping download")
        else:
            # Download
            if not download_file(url, tar_path):
                print(f"⚠️  Failed to download {name}, continuing...")
                continue
        
        # Extract
        if not extract_tar(tar_path, data_dir):
            print(f"⚠️  Failed to extract {name}, continuing...")
            continue
        
        # Optionally remove tar file to save space
        # tar_path.unlink()
        # print(f"🗑️  Removed {tar_filename} to save space")
    
    # Verify dataset structure
    print("\n" + "=" * 70)
    print("🔍 Verifying dataset structure...")
    print("=" * 70)
    
    expected_paths = [
        voc_dir / 'VOC2007',
        voc_dir / 'VOC2012',
    ]
    
    for path in expected_paths:
        if path.exists():
            # Count images
            img_dir = path / 'JPEGImages'
            if img_dir.exists():
                num_images = len(list(img_dir.glob('*.jpg')))
                print(f"✅ {path.name}: {num_images} images")
            else:
                print(f"⚠️  {path.name}: JPEGImages directory not found")
        else:
            print(f"❌ {path.name}: Not found")
    
    print("\n" + "=" * 70)
    print("✅ Dataset download complete!")
    print("=" * 70)
    print(f"\n📊 Dataset statistics:")
    print(f"   - VOC2007 trainval: ~5,011 images")
    print(f"   - VOC2007 test: ~4,952 images")
    print(f"   - VOC2012 trainval: ~11,540 images")
    print(f"   - Total: ~21,503 images")
    print(f"\n🏷️  Classes (20): aeroplane, bicycle, bird, boat, bottle, bus, car,")
    print(f"   cat, chair, cow, diningtable, dog, horse, motorbike, person,")
    print(f"   pottedplant, sheep, sofa, train, tvmonitor")
    print("\n📝 Next steps:")
    print("   1. Run: python src/train.py --data data/voc.yaml")
    print("   2. Check experiments/configs/ for training configurations")

if __name__ == '__main__':
    main()

"""
Pascal VOC Dataset loader for YOLOv7 training.
Handles VOC XML annotations and converts to YOLO format.
"""

import torch
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VOCDataset(data.Dataset):
    """Pascal VOC dataset for object detection."""
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root_dir: str,
        year: str = '2007',
        image_set: str = 'trainval',
        img_size: int = 640,
        augment: bool = True
    ):
        """
        Args:
            root_dir: Path to VOCdevkit directory
            year: VOC year ('2007' or '2012')
            image_set: 'train', 'val', 'trainval', or 'test'
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.root = Path(root_dir)
        self.year = year
        self.image_set = image_set
        self.img_size = img_size
        self.augment = augment
        
        # Paths
        self.voc_root = self.root / f'VOC{year}'
        self.img_dir = self.voc_root / 'JPEGImages'
        self.anno_dir = self.voc_root / 'Annotations'
        
        # Load image IDs
        image_set_file = self.voc_root / 'ImageSets' / 'Main' / f'{image_set}.txt'
        with open(image_set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}
        
        # Data augmentation
        self.transforms = self._get_transforms()
        
        print(f"Loaded {len(self.image_ids)} images from VOC{year} {image_set}")
    
    def _get_transforms(self):
        """Get data augmentation transforms."""
        if self.augment:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels']
            ))
    
    def _parse_annotation(self, anno_path: Path) -> Tuple[List, List]:
        """Parse VOC XML annotation."""
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            difficult = int(obj.find('difficult').text)
            if difficult:
                continue
            
            cls_name = obj.find('name').text
            if cls_name not in self.class_to_idx:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
        
        return boxes, labels
    
    def __getitem__(self, index: int):
        """Get image and target."""
        img_id = self.image_ids[index]
        
        # Load image
        img_path = self.img_dir / f'{img_id}.jpg'
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        anno_path = self.anno_dir / f'{img_id}.xml'
        boxes, labels = self._parse_annotation(anno_path)
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            # No objects, just transform image
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = []
            labels = []
        
        # Convert to tensor format
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            # Convert from xyxy to xywh normalized
            img_h, img_w = self.img_size, self.img_size
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w  # x center
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h  # y center
            boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) / img_w  # width
            boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) / img_h  # height
            
            # Combine boxes and labels
            targets = torch.cat([
                labels.unsqueeze(1),
                boxes
            ], dim=1)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, targets, img_id
    
    def __len__(self):
        return len(self.image_ids)


def collate_fn(batch):
    """Custom collate function for variable number of objects."""
    images, targets, img_ids = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Add batch index to targets
    for i, target in enumerate(targets):
        if len(target) > 0:
            target = torch.cat([
                torch.full((len(target), 1), i),
                target
            ], dim=1)
            targets[i] = target
    
    # Concatenate all targets
    targets = torch.cat([t for t in targets if len(t) > 0], dim=0)
    
    return images, targets, img_ids


def create_dataloaders(
    data_config: Dict,
    batch_size: int = 16,
    img_size: int = 640,
    workers: int = 4
):
    """
    Create train and validation dataloaders.
    
    Args:
        data_config: Dict with 'path', 'train', 'val' keys
        batch_size: Batch size
        img_size: Image size
        workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader
    """
    root_dir = data_config['path']
    
    # Create datasets
    # Combine VOC2007 and VOC2012 trainval for training
    train_dataset_07 = VOCDataset(
        root_dir=root_dir,
        year='2007',
        image_set='trainval',
        img_size=img_size,
        augment=True
    )
    
    train_dataset_12 = VOCDataset(
        root_dir=root_dir,
        year='2012',
        image_set='trainval',
        img_size=img_size,
        augment=True
    )
    
    # Combine datasets
    train_dataset = data.ConcatDataset([train_dataset_07, train_dataset_12])
    
    # VOC2007 test for validation
    val_dataset = VOCDataset(
        root_dir=root_dir,
        year='2007',
        image_set='test',
        img_size=img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    print("Testing VOC Dataset loader...")
    
    # Test dataset
    dataset = VOCDataset(
        root_dir='./data/VOCdevkit',
        year='2007',
        image_set='trainval',
        img_size=640,
        augment=True
    )
    
    print(f"\n✅ Dataset size: {len(dataset)}")
    print(f"✅ Classes: {len(dataset.VOC_CLASSES)}")
    
    # Test loading one sample
    image, targets, img_id = dataset[0]
    print(f"\n✅ Sample loaded:")
    print(f"   Image shape: {image.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Image ID: {img_id}")
    print(f"   Number of objects: {len(targets)}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    images, targets, img_ids = next(iter(loader))
    print(f"\n✅ Batch loaded:")
    print(f"   Images shape: {images.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Batch size: {len(img_ids)}")
    
    print("\n✅ Dataset tests passed!")

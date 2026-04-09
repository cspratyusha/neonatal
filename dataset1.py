import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class IcopeDataset(Dataset):
    def __init__(self, root_dir, sequence_length=16, train=True):
        self.root_dir        = root_dir
        self.sequence_length = sequence_length
        self.transform       = get_transforms(train)
        self.samples         = []

        for label in ["pain", "no_pain"]:
            class_dir = os.path.join(root_dir, label)
            if not os.path.exists(class_dir):
                print(f"⚠️  Folder missing: {class_dir}")
                continue
            label_value = 1 if label == "pain" else 0
            for video_folder in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_folder)
                if not os.path.isdir(video_path):
                    continue
                frames = sorted([
                    f for f in os.listdir(video_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                if len(frames) >= sequence_length:
                    self.samples.append((video_path, label_value, frames))

        pain_count    = sum(1 for _, l, _ in self.samples if l == 1)
        no_pain_count = sum(1 for _, l, _ in self.samples if l == 0)
        print(f"✅ Dataset loaded: {len(self.samples)} samples "
              f"(pain={pain_count}, no_pain={no_pain_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, frames = self.samples[idx]
        total = len(frames)

        # Uniform sampling across full video — better than always taking first N frames
        indices  = [int(i * total / self.sequence_length) for i in range(self.sequence_length)]
        selected = [frames[i] for i in indices]

        sequence = []
        for frame_name in selected:
            frame_path = os.path.join(video_path, frame_name)
            img = cv2.imread(frame_path)
            if img is None:
                # Fallback: use a black frame if file is corrupt
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # ← critical fix
            img = self.transform(img)
            sequence.append(img)

        return torch.stack(sequence), torch.tensor(label, dtype=torch.long)


class ValTestDataset(Dataset):
    """
    Wraps a Subset so that val/test always get inference-only transforms
    (no augmentation), regardless of what the parent dataset uses.
    """
    def __init__(self, subset, sequence_length=16):
        self.subset          = subset
        self.sequence_length = sequence_length
        self.transform       = get_transforms(train=False)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        video_path, label, frames = self.subset.dataset.samples[self.subset.indices[idx]]
        total    = len(frames)
        indices  = [int(i * total / self.sequence_length) for i in range(self.sequence_length)]
        selected = [frames[i] for i in indices]

        sequence = []
        for frame_name in selected:
            frame_path = os.path.join(video_path, frame_name)
            img = cv2.imread(frame_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            sequence.append(img)

        return torch.stack(sequence), torch.tensor(label, dtype=torch.long)
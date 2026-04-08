import os
import torch
from torch.utils.data import Dataset
import cv2

class IcopeDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.samples = []

        for label in ["pain", "no_pain"]:
            class_dir = os.path.join(root_dir, label)
            label_value = 1 if label == "pain" else 0

            for video_folder in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_folder)
                frames = sorted(os.listdir(video_path))

                if len(frames) >= sequence_length:
                    self.samples.append((video_path, label_value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sorted(os.listdir(video_path))[:self.sequence_length]

        sequence = []

        for frame_name in frames:
            frame_path = os.path.join(video_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = torch.tensor(img).permute(2, 0, 1).float()
            sequence.append(img)

        sequence = torch.stack(sequence)
        return sequence, torch.tensor(label)

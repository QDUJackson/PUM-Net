import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalRegistrationDataset(Dataset):
    def __init__(self, model1_dir, model2_dir, label_dir,seg_dir ,model1_normal_dir,transform=None):
        self.model1_dir = model1_dir
        self.model2_dir = model2_dir
        self.label_dir = label_dir
        self.seg_dir = seg_dir
        self.model1_normal_dir = model1_normal_dir

        self.transform = transform

        self.model1_files = sorted(os.listdir(model1_dir))
        self.model2_files = sorted(os.listdir(model2_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.seg_files = sorted(os.listdir(seg_dir))
        self.model1_normal_files = sorted(os.listdir(model1_normal_dir))

    def __len__(self):
        return len(self.model1_files)

    def __getitem__(self, idx):

        model1_path = os.path.join(self.model1_dir, self.model1_files[idx])
        model2_path = os.path.join(self.model2_dir, self.model2_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        seg_path = os.path.join(self.seg_dir, self.seg_files[idx])
        model1_normal_path = os.path.join(self.model1_normal_dir, self.model1_normal_files[idx])

        model1_image = cv2.imread(model1_path, cv2.IMREAD_GRAYSCALE)
        model2_image = cv2.imread(model2_path, cv2.IMREAD_GRAYSCALE)
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        seg_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        model1_normal_image = cv2.imread(model1_normal_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            model1_image = self.transform(model1_image)
            model2_image = self.transform(model2_image)
            label_image = self.transform(label_image)
            seg_image = self.transform(seg_image)
            model1_normal_image = self.transform(model1_normal_image)


        model1_image = model1_image.to(dtype=torch.float32)
        model2_image = model2_image.to(dtype=torch.float32)
        label_image = label_image.to(dtype=torch.float32)
        seg_image = seg_image.to(dtype=torch.float32)
        model1_normal_image = model1_normal_image.to(dtype=torch.float32)

        return {
            'model1': model1_image,
            'model2': model2_image,
            'label': label_image,
            'seg': seg_image,
            'model_normal': model1_normal_image
        }

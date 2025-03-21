import os
import torch

from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class USKCoffeeDataset(Dataset):
    def __init__(self, phase="train", transform=None):
        self.root_path = "D:/Research/Dataset/USK-Coffee"
        assert phase in ["train", "test", "val"]
        
        # populate data
        self.data = glob(os.path.join(self.root_path, phase, "*", "*.jpg"))
        # pre-populate labels
        self.labels = os.listdir(os.path.join(self.root_path, phase))
        print(f"Labels: {self.labels}")
        
        # if no transform is defined, use this default transform toTensor
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        
    # Handy utility function to get the label from the image path (on platform windows)       
    def get_label(self, image_path):
        return image_path.split(os.sep)[-2]
    
    def label_to_index(self, label):
        return self.labels.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        label = self.get_label(image_path)
        label = self.label_to_index(label)
        
        # if transform is defined, apply it
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
class USKCoffeeDatasetDefect(Dataset): # zero label is defect
    def __init__(self, phase="train", transform=None):
        self.root_path = "D:/Research/Dataset/USK-Coffee"
        assert phase in ["train", "test", "val"]
        
        # populate data
        self.data = glob(os.path.join(self.root_path, phase, "*", "*.jpg"))
        # pre-populate labels
        self.labels = os.listdir(os.path.join(self.root_path, phase))
        # Labels: ['defect', 'longberry', 'peaberry', 'premium']
        
        # if no transform is defined, use this default transform toTensor
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        
    # Handy utility function to get the label from the image path (on platform windows)       
    def get_label(self, image_path):
        return image_path.split(os.sep)[-2]
    
    def label_to_index(self, label):
        # Labels: ['defect', 'longberry', 'peaberry', 'premium']
        # we want to convert 'defect' to 0, 'longberry' to 1, 'peaberry' to 1, 'premium' to 1, label 1 means good coffee
        if label == "defect":
            return 0
        else:
            return 1 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        label = self.get_label(image_path)
        label = self.label_to_index(label)
        
        # if transform is defined, apply it
        if self.transform:
            image = self.transform(image)
        
        return image, label
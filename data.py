import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        """
        Finds the class folders in a dataset.
        Ensures that hidden files/directories like .DS_Store are ignored.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and not entry.name.startswith('.'))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class FruitsDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform=None, augmented_transform=None, target_transform=None, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.augmented_transform = augmented_transform or transform
        self.target_transform=target_transform
        self.num_workers = num_workers
        
        self.persistent_workers = True if num_workers else False
        self.pin_memory = torch.cuda.is_available()
        
        self.train_dir = os.path.join(data_dir, "train")
        self.valid_dir = os.path.join(data_dir, "valid")
        self.test_dir = os.path.join(data_dir, "test")

#     def setup(self, stage=None):
        self.train_set = CustomImageFolder(root=self.train_dir, transform=self.augmented_transform, target_transform=self.target_transform)
        self.valid_set = CustomImageFolder(root=self.valid_dir, transform=self.transform, target_transform=self.target_transform)
        self.test_set = CustomImageFolder(root=self.test_dir, transform=self.transform, target_transform=self.target_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
import os
import torch
import lightning as L
from torchvision import transforms
from lightning.pytorch.loggers import TensorBoardLogger
from config import *
from data import FruitsDataModule
from model import FineTuneEfficientNet

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device = torch.device(device)

print(f"Device: {device}, Data Dir: {DATA_DIR}, Num Classes: {NUM_CLASSES}")


augmented_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])


transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

dm = FruitsDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, transform=transform, augmented_transform=augmented_transform, num_workers=NUM_WORKERS, target_transform=None)

effnet_model = FineTuneEfficientNet(num_classes=NUM_CLASSES, dropout=0.2, learning_rate=0.001)

logger = TensorBoardLogger("tb_logs", name="fruit_model")

trainer = L.Trainer(max_epochs=MAX_EPOCHS, devices=-1, precision="16-mixed", deterministic=True, logger=logger)
trainer.fit(effnet_model, datamodule=dm)
trainer.test(effnet_model, datamodule=dm)
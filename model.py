import lightning as L
# from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models
from torch import nn
import torch

class FineTuneEfficientNet(L.LightningModule):
    def __init__(self, num_classes=5, dropout=0.2, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.backbone = models.efficientnet_v2_s(weights="DEFAULT")
        
        for param in list(self.backbone.parameters()):
            param.requires_grad = False
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Identity()  
        
        dim=128
        self.model = nn.Sequential(
            self.backbone,
            nn.Linear(num_features, dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        train_acc = accuracy_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        train_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_precision', train_precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_recall', train_recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1', train_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        
        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        
        val_acc = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        val_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_precision', val_precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_recall', val_recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1', val_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        
        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        
        test_acc = accuracy_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        test_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        test_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
        
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_precision', test_precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_recall', test_recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_f1', test_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

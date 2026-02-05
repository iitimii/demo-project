BATCH_SIZE = 128
NUM_WORKERS = 0
MAX_EPOCHS = 30
LEARNING_RATE = 1e-2
DROPOUT = 0.3
LAST_LAYER = -2
DATA_DIR = "Fruits Classification"

import os
NUM_CLASSES = len([d for d in os.listdir(os.path.join(DATA_DIR, "train")) if not d.startswith('.')])
CLASS_NAMES = sorted([d for d in os.listdir(os.path.join(DATA_DIR, "train")) if not d.startswith('.')])
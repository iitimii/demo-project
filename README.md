# Fruit Classification

End-to-end fruit image classification using transfer learning (EfficientNetV2-S) with PyTorch Lightning, served via FastAPI with a simple web frontend.

## Project Structure

- `app.py`: FastAPI server and API endpoints
- `model.py`: PyTorch Lightning model definition
- `data.py`: Data loading logic and `FruitsDataModule`
- `train.py`: Training and evaluation script
- `config.py`: Global configuration (batch size, epochs, paths)
- `frontend/`: Web UI (HTML, CSS, JavaScript)
- `Fruits Classification/`: Dataset directory (train/valid/test)
- `model.ckpt`: Fine-tuned model checkpoint

## Setup

```bash
git clone https://github.com/iitimii/demo-project.git
cd fruits-classification
pip install -r requirements.txt
```

## Usage

### Training the Model:

```bash
python train.py
```

### Running the Web Application:

```bash
python app.py
```

If browser window doesn't pop up, the application will be available at `http://127.0.0.1:8000`.

## Monitoring

### View TensorBoard logs:

```bash
tensorboard --logdir tb_logs
```

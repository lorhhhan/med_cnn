# Intelligent Medical Image Classification and Consultation System

This project implements a lightweight intelligent medical consultation platform. It integrates disease image classification, uncertainty-aware fallback to vision-language models (MiniGPT-4), and optional integration with language models like ChatGLM or Vicuna-7B for follow-up consultation.

---

##  Setup Instructions

### 1. Clone and Replace MiniGPT-4

Clone the official MiniGPT-4 repo:

git clone https://github.com/Vision-CAIR/MiniGPT-4.git


Then, **replace its contents** with the customized files in this project's `MiniGPT-4/` folder.

- `minigpt4_infer.py` is the main entry point for vision-language inference and is called by the classification logic when model confidence is low.

### 2. Vicuna-7B Model Weights

Place the downloaded **Vicuna-7B model and tokenizer files** inside the `vicuna-7b/` directory. The structure should follow HuggingFace's format and support loading via Transformers.
---

## Project Structure
```
med/
├── checkpoints/        # Trained CNN weights (e.g., efficientnet_b4.pth)
├── images/             # Saved outputs if needed
├── MiniGPT-4/          # Modified MiniGPT-4 inference module
├── test_img/           # Test images for classification and vision tasks
├── venv/               # Local virtual environment
├── vicuna-7b/          # Vicuna-7B model files
├── config.py
├── dataset.py
├── model.py            # EfficientNet-B4 architecture
├── predict.py          # Unified prediction function
├── preprocs.py         # Dataset preprocessing
├── requirements.txt
├── train.py            # Training script
└── utils.py
```
---

##  Usage

### Classification + Vision-Language Description
```bash
python predict.py
```
If the classification model has low confidence, it calls MiniGPT-4 for image description.
---

##  Training

Train the image classifier with EfficientNet-B4:
```bash
python train.py
```
Ensure training images and labels are correctly loaded in `dataset.py`.
---

##  Dependencies

Install required packages:

```bash
pip install -r requirements.txt
pip install sentencepiece accelerate bitsandbytes
```
---

##  Data Sources

- HAM10000 (ISIC Challenge)
- Roboflow datasets (e.g., Conjunctivitis, Eczema, etc.)
---

##  Note

- You must prepare your Vicuna-7B weights and tokenizer separately.
- MiniGPT-4 model must be pre-downloaded or built according to their documentation.

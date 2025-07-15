import os

# BASE_DIR = r"D:\codeSW\Python\med"
# DATA_DIR = os.path.join(BASE_DIR, "images")

BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, "images")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "efficientnet_b4.pth")
CLASS_NAMES = [
    'Acne',
    'Conjunctivitis',
    'Eczema',
    'Urticaria',
    'Cataracts',
    'Glaucoma',
    'UVeitis'
]

class TrainConfig:

    num_epochs = 12
    batch_size = 8
    learning_rate = 5e-5
    image_size = 380
    num_workers = 2
    weight_decay = 0.01
    warmup_ratio = 0.05
    gradient_accumulation_steps = 2
    lr_scheduler_type = "cosine"
    fp16 = True

greater_is_better=True,


# BASE_DIR = r"D:\codeSW\Python\med"
# DATA_DIR = os.path.join(BASE_DIR, "images")
# CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "efficientnet_b4.pth")
# CLASS_NAMES = ['Acne', 'Eczema', 'Enfeksiyonel', 'Urticaria','Conjunctivitis']
#
# class TrainConfig:
#     num_epochs = 1
#     batch_size = 8
#     learning_rate = 1e-4
#     image_size = 380
#     num_workers = 0
#     weight_decay = 0.01
#     warmup_ratio = 0.1,
#     gradient_accumulation_steps = 4
#     lr_scheduler_type = "cosine_with_restarts"
#     fp16 = False  # 若使用 AMP，可改为 True
# greater_is_better=True,
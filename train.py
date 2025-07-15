import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import build_model
from dataset import get_data_loaders
from config import DATA_DIR, CHECKPOINT_PATH, CLASS_NAMES, TrainConfig, greater_is_better
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train():
    cfg = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, class_names, all_targets = get_data_loaders(
        DATA_DIR, cfg.batch_size, cfg.image_size, cfg.num_workers
    )

    model = build_model(len(class_names)).to(device)

    # 用训练集中真实出现的标签构建 class_weight
    train_labels = [label for _, label in train_loader.dataset]

    # 获取唯一类
    unique_classes = np.unique(train_labels)

    # 自动对真实使用的标签编号做平衡
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_score = -float('inf') if greater_is_better else float('inf')

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_loader.dataset)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        tqdm.write(f"[Epoch {epoch+1}] Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | F1 = {val_f1:.4f}")

        score = val_f1
        is_better = score > best_score if greater_is_better else score < best_score
        if is_better:
            best_score = score
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            tqdm.write(f" [Epoch {epoch+1}] New best model saved! Val F1 = {val_f1:.4f}")

if __name__ == '__main__':
    train()



# def train():
#     cfg = TrainConfig()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 加载数据和模型
#     loader, class_names = get_data_loaders(DATA_DIR, cfg.batch_size, cfg.image_size, cfg.num_workers)
#     model = build_model(len(class_names)).to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
#
#     for epoch in range(cfg.num_epochs):
#         model.train()
#         total_loss = 0.0
#         total_correct = 0
#
#         progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
#         for images, labels in progress_bar:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_correct += (outputs.argmax(1) == labels).sum().item()
#             progress_bar.set_postfix(loss=loss.item())
#
#         acc = total_correct / len(loader.dataset)
#         tqdm.write(f"[Epoch {epoch + 1}] Total Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")
#
#
#     os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
#
#     torch.save(model.state_dict(), CHECKPOINT_PATH)
#     tqdm.write(f"Model saved to: {CHECKPOINT_PATH}")
#
#
# if __name__ == '__main__':
#     train()

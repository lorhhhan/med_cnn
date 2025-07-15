
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import CLASS_NAMES

def get_data_loaders(data_dir, batch_size=16, image_size=380, num_workers=0, val_split=0.2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    dataset.samples = [s for s in dataset.samples if os.path.basename(os.path.dirname(s[0])) in CLASS_NAMES]
    dataset.targets = [dataset.class_to_idx[os.path.basename(os.path.dirname(s[0]))] for s in dataset.samples]

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 替换验证集 transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dataset.classes, dataset.targets



# def get_data_loaders(data_dir, batch_size, image_size, num_workers=0):
#     from torchvision import datasets, transforms
#     from torch.utils.data import DataLoader
#
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3),
#     ])
#
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return loader, dataset.classes


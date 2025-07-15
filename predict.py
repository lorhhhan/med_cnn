import torch
from PIL import Image
from torchvision import transforms
from model import build_model

from minigpt4 import MiniGPT4

#git clone https://github.com/Vision-CAIR/MiniGPT-4.git
#cd MiniGPT-4
#
minigpt_model = MiniGPT4()
minigpt_model.load_pretrained_weights("model/path")
minigpt_model.eval()

def description_minigpt(image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    description = minigpt_model.generate(img, prompt)
    return description

def predict(image_path, model_path, class_names, threshold, prompt="Describe this image as detailed as possible, including color, shape, and distribution."):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().squeeze()

    # 输出详细概率表
    print("----- 概率分布 -----")
    for i, prob in enumerate(probs):
        print(f"{class_names[i]}: {prob.item():.3f}")
    print("--------------------")

    max_prob, pred_class = torch.max(probs, dim=0)

    if max_prob.item() < threshold:
        # MiniGPT-4
        description = description_minigpt(image_path, prompt)
        return "other", max_prob.item(), description
    else:
        return class_names[pred_class.item()], max_prob.item(), None


if __name__ == '__main__':
    class_names = ['Acne', 'Conjunctivitis', 'Eczema', 'Urticaria', 'Cataracts', 'Glaucoma', 'UVeitis']
    image_path = 'test_image.jpg'
    model_path = 'checkpoints/efficientnet_b4.pth'

    result, prob, description = predict(
        image_path,
        model_path,
        class_names,
        threshold=0.8,
        prompt="Describe this image as detailed as possible, including color, shape, and surface texture. Write in multiple sentences."
    )

    if result == "other":
        print(f"result: {result} (prob: {prob:.2f})")
        print(f"MiniGPT-4 描述: {description}")
    else:
        print(f"result: {result} (prob: {prob:.2f})")


















# def predict(image_path, model_path, class_names, threshold=0.5, show_table=True):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = build_model(len(class_names))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     transform = transforms.Compose([
#         transforms.Resize((380, 380)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(input_tensor)
#         probs = torch.softmax(output, dim=1).cpu().squeeze()
#
#     # 输出详细概率表
#     if show_table:
#         print("----- 概率分布 -----")
#         for i, prob in enumerate(probs):
#             print(f"{class_names[i]}: {prob.item():.3f}")
#         print("--------------------")
#
#     max_prob, pred_class = torch.max(probs, dim=0)
#
#     if max_prob.item() < threshold:
#         return "other", max_prob.item()
#     else:
#         return class_names[pred_class.item()], max_prob.item()
#
# if __name__ == '__main__':
#     class_names = ['Acne', 'Conjunctivitis', 'Eczema', 'Urticaria', 'Cataracts', 'Glaucoma', 'UVeitis']
#     image_path = 'test_img/222.jpg'
#     model_path = 'checkpoints/efficientnet_b4.pth'
#     result, prob = predict(image_path, model_path, class_names, threshold=0.7, show_table=True)
#     print(f"recognition results: {result} (置信度: {prob:.2f})")




# def predict(image_path, model_path, class_names):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = build_model(len(class_names))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#
#     transform = transforms.Compose([
#         transforms.Resize((380, 380)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_class = output.argmax(1).item()
#
#     return class_names[pred_class]
#
# # 示例调用
# if __name__ == '__main__':
#     # class_names = ['Acne', 'Conjunctivitis', 'Eczema', 'Enfeksiyonel', 'other', 'urticaria']
#     class_names = ['Acne', 'Cataracts', 'Conjunctivitis', 'Eczema', 'Glaucoma', 'UVeitis', 'urticaria']
#     image_path = 'test_acne.jpg'
#     model_path = 'checkpoints/efficientnet_b4.pth'
#     result = predict(image_path, model_path, class_names)
#     print(f"—————————————————————————————————")
#     print(f"recognition results: {result}")
#     print(f"—————————————————————————————————")
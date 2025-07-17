import torch
from PIL import Image
from torchvision import transforms
from model import build_model
from subprocess import check_output

import os

def predict(image_path, model_path, class_names, threshold=0.8, prompt="A detailed description of this skin lesion."):
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

    print("Prob Distribution:")
    for i, prob in enumerate(probs):
        print(f"{class_names[i]}: {prob.item():.3f}")
    print("------------------------------------")

    max_prob, pred_class = torch.max(probs, dim=0)

    if max_prob.item() >= threshold:
        return class_names[pred_class.item()], max_prob.item(), None

    #MiniGPT-4
    print("MiniGPT-4...")

    result = check_output([
        "python",
        "MiniGPT-4/minigpt4_infer.py",
        "--cfg-path", "MiniGPT-4/eval_configs/minigpt4_eval.yaml",
        "--gpu-id", "0",
        "--img-path", image_path,
        "--prompt", prompt
    ], text=True)

    # 提取生成文本
    description = result.split("Description :")[-1].strip()
    return "other", max_prob.item(), description



if __name__ == '__main__':
    class_names = ['Acne', 'Conjunctivitis', 'Eczema', 'Urticaria', 'Cataracts', 'Glaucoma', 'UVeitis']
    image_path = 'test_img/t1.jpg'
    model_path = 'checkpoints/efficientnet_b4.pth'

    result, prob, description = predict(
        image_path,
        model_path,
        class_names,
        threshold=0.8,
        prompt="Describe this image, including shape, color, borders, and surface texture. Write as if explaining to a medical student."
    )

    print(f"Prediction: {result} (confidence: {prob:.2f})")
    if description:
        print("\nDescription:\n" + description)





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
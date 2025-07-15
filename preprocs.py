import os
from PIL import Image


def convert_to_jpg(input_folder):
    # 遍历目录中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 跳过非文件项和已为jpg的文件
        if not os.path.isfile(input_path) or filename.lower().endswith('.jpg'):
            continue

        try:
            # 打开图像文件
            with Image.open(input_path) as img:
                # 创建输出文件名(保持原名但改扩展名)
                output_filename = os.path.splitext(filename)[0] + '.jpg'
                output_path = os.path.join(input_folder, output_filename)

                # 转换为RGB模式(防止RGBA出现问题)并保存为JPG
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)

                # 可选: 删除原始文件
                os.remove(input_path)
                # print(f"转换成功: {filename} -> {output_filename}")

        except Exception as e:
            print(f"无法转换 {filename}: {str(e)}")


# 使用示例
acne_folder = r'D:\codeSW\Python\med\images\Acne'
convert_to_jpg(acne_folder)
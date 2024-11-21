import cv2
import numpy as np

def split_image_with_white_borders(image_path, rows, cols, border_width=1):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算每个小块的高度和宽度
    block_height = height // rows
    block_width = width // cols

    # 创建一个新的空白图像，用于存储带有边界的分割块
    new_height = block_height * rows + border_width * (rows - 1)
    new_width = block_width * cols + border_width * (cols - 1)
    result_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    # 将每个小块放入新图像中
    for i in range(rows):
        for j in range(cols):
            # 计算当前小块的起始位置
            start_y = i * (block_height + border_width)
            start_x = j * (block_width + border_width)

            # 计算当前小块的结束位置
            end_y = start_y + block_height
            end_x = start_x + block_width

            # 将当前小块复制到新图像中
            result_image[start_y:end_y, start_x:end_x] = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]

    return result_image

# 参数设置
image_path = 'sm.png'  # 替换为你的图片路径
rows = 5
cols = 6
border_width = 2

# 分割图像
result_image = split_image_with_white_borders(image_path, rows, cols, border_width)

# 保存结果图像
output_path = 'out.png'
cv2.imwrite(output_path, result_image)

print(f"Image saved to {output_path}")
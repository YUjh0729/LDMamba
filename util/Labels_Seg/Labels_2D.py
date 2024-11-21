from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def apply_color_labels(image_path, output_path):
    # 打开图片并转换为灰度图像
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 获取唯一的像素值
    unique_values = np.unique(img_array)

    # 创建颜色映射（根据像素值数量生成不同的颜色）
    cmap = plt.get_cmap('tab20', len(unique_values) - 1)  # 不包含黑色
    color_img = np.zeros((*img_array.shape, 3))

    # 遍历唯一的像素值，为每个像素值分配颜色
    for idx, val in enumerate(unique_values):
        if val == 0:
            # 保持黑色像素不变
            color_img[img_array == val] = [0, 0, 0]  # 黑色
        else:
            # 为非黑色像素分配颜色
            mask = (img_array == val)
            color_img[mask] = cmap(idx - 1)[:3]  # 获取对应的RGB值，-1是为了避免黑色的索引

    # 保存新图片
    plt.imsave(output_path, color_img)

    # 显示结果
    plt.imshow(color_img)
    plt.axis('off')
    plt.show()


# 示例调用
input_image = 'yjh/GT.png'  # 输入图片路径
output_image = 'yjh/res/GT.png'  # 输出图片路径
apply_color_labels(input_image, output_image)
print("输出成功")
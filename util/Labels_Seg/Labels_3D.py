import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_color_mapping(unique_values):
    # 创建颜色映射
    color_mapping = {}
    # 使用不同的颜色为不同的值分配颜色
    colors = plt.get_cmap('tab20', len(unique_values))  # 使用 colormap

    for idx, value in enumerate(unique_values):
        color_mapping[value] = colors(idx)[:3]  # 只取 RGB，不取 Alpha

    return color_mapping


def display_3d_nifti_with_colors(nifti_file, output_file, elevation=30, azimuth=30):
    # 读取 NIfTI 文件
    img = nib.load(nifti_file)
    img_data = img.get_fdata()

    # 获取唯一的像素值
    unique_values = np.unique(img_data)

    # 创建颜色映射
    color_mapping = create_color_mapping(unique_values)

    # 获取 3D 网格的坐标
    x, y, z = np.indices(img_data.shape)

    # 绘制 3D 图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置背景为黑色
    fig.patch.set_facecolor('black')  # 设置图形背景为黑色
    ax.set_facecolor('black')  # 设置坐标轴背景为黑色

    # 遍历每个像素点
    for value in unique_values:
        if value == 0:  # 如果值为 0（背景），可以选择跳过
            continue
        # 获取当前值的掩膜
        mask = img_data == value

        # 获取对应的坐标
        ax.scatter(x[mask], y[mask], z[mask], color=color_mapping[value], alpha=0.5)

    # 去除坐标轴和边框
    ax.set_axis_off()

    # 设置视角
    ax.view_init(elev=elevation, azim=azimuth)  # 自定义视角

    # 保存图像到文件
    plt.savefig(output_file, bbox_inches='tight', facecolor='black')  # 保存为 PNG 文件，背景黑色
    plt.close(fig)  # 关闭图形，释放内存


# 示例调用
nifti_file = 'yjh/TW_Attn.nii.gz'  # NIfTI 文件路径
output_file = 'yjh/res/TW_Attn.png'  # 输出的图像文件路径
display_3d_nifti_with_colors(nifti_file, output_file, elevation=0, azimuth=90)  # 自定义视角
print("输出成功")

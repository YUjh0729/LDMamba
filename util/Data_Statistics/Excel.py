import pandas as pd

# 读取无后缀文件
file_path = 'Data/3D/CT/nnUNetTrainerTDMambaNet__nnUNetPlans__3d/DeformConv55/evaluation_result_DSC'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    data = file.read()

# 将数据分割成行
lines = data.strip().split('\n')

# 将每行分割成列
rows = [line.split(',') for line in lines]

# 创建 DataFrame
df = pd.DataFrame(rows[1:], columns=rows[0])

# 保存为 Excel 文件
df.to_excel('output.xlsx', index=False)

print("数据已成功保存到 output.xlsx")
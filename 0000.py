import matplotlib.pyplot as plt
import numpy as np

# 创建一个示例数据
data = np.random.rand(10, 10)
data[data<0.2] = 0
data[data>0.8] = 1
plt.rcParams['font.family'] = 'Times New Roman'


# 创建一个隐藏轴的图像
fig, ax = plt.subplots()

# 添加颜色条
cax = ax.imshow(data, cmap='coolwarm')
cbar = fig.colorbar(cax, orientation='horizontal')

# 隐藏主轴
ax.remove()

# 获取数据的最小值和最大值
# data_min, data_max = data.min(), data.max()

# 设置颜色条的刻度为最大值和最小值
# ax.set_xticks([0,0.2,0,4,0.6,0.8,1.0])
# cbar.set_ticklabels([f'{data_min:.2f}', f'{data_max:.2f}'])
# 显示颜色条
plt.savefig("colorbar.png",dpi=200)
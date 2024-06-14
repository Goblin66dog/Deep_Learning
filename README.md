# Deep Learning
 深度学习算法总结（Personal Configuration）

## 目录：
- [项目の目的](#项目の目的)
- [项目の结构](#项目の结构)
- [项目の详述](#项目の详述)
  - [训练准备](#训练准备)
    - [数据读取](#数据读取) 
    - [数据处理](#数据处理)
    - [网络库](#网络库)
  - [训练配置](#训练配置)
    - [数据分配](#数据分配)
    - [数据加载](#数据加载)
    - [损失函数](#损失函数)
    - [优化器](#优化器)
    - [训练策略](#训练策略)
  - [模型部署（后话）](#模型部署)
- [Thanks For Supporting!](#Thanks For Supporting!)
- [遭不住了有点多，哥们要考研了，开摆!(2024.4.30)](#遭不住了有点多，哥们要考研了，开摆!(2024.4.30))

# 项目の目的

# 项目の结构
<pre> Datasets
->数据读取
->数据处理
->数据分配
->网络训练<-训练策略(损失函数+优化器)
->输出
</pre>

# 项目の详述
## 训练准备
### 数据读取
**_Data_Reader_**
- **数据读取模块**
- 用于各种栅格数据读取
  - 读取包括.TIF/.PNG/.JPG的各种图像
  - 返回的栅格类型为：C·H·W
- 代码更加精简
<pre>
- Dataset          : 定义了一个Dataset类
  - input_file_path: 输入文件路径
  - self.data      : gdal对象
  - self.width     : 图像宽
  - self.height    : 图像高
  - self.proj      : 地图投影信息
  - self.geotrans  : 仿射变换参数
  - self.array     : 图像数据
  - self.bands     : 图像波段（通道数）
  - self.type      : 图像数据格式
- 无返回值
</pre>
### 数据处理
**_Padding_**
- **对单张图像进行Padding操作**
<pre>
- Padding           : 定义了一个Padding类
  - image           : 输入图像栅格
  - image_shape     : N·C·H·W的顺序
  - self.mir        : 镜像Padding操作
    - target_height : Padding目标高 
    - target_width  : Padding目标宽
  - self.nor        : 常规Padding操作（直接补0）
    - target_height : Padding目标高 
    - target_width  : Padding目标宽
  - self.min        : 最小Padding操作（用于适应多层下采样或卷积）
    - divide        : 需要适应的被除数
    - 函数返回为输入图像与divide之间的最小公倍数大小的图像
- 返回一个经Padding操作的图像
</pre>
**_Random_Flip_**
- **对原始图像和标签图像进行随机反转操作**
<pre>
- RandomFlip        : 定义了一个RandomFlip类
  - image           : 输入原始图像栅格
  - label           : 输入标签图像栅格
  - self.random_flip     : 随机翻转
- 返回一对经随机翻转的图像元组
</pre>
**_Flip8x_**
- **对图像进行包括翻转、旋转在内的所有角度翻转操作**
<pre>
- Flip8x        : 定义了一个Flip8x类
  - image           : 输入图像栅格
  - image_shape     : N·C·H·W的顺序
  - self.flip8x     : 8种角度翻转
    - 函数返回的是8种角度翻转的列表栅格
  - self.flip       : 返回8种角度反转的图像，保存到本地
</pre>
**_Percent_Linear_Enhancement_**
- **线性拉伸**

**_DataPreProcessor_**
- **对图像进行预处理操作，并将图像保存到本地**
<pre>
- Processor              : 定义了一个Processor类
  - self.batch_processor : 数据批处理
    - input_pack_path    : 输入数据集路径
    - output_pack_path   : 输出文件夹名称
    - image_shape        : N·C·H·W的顺序
    - mode               : 处理模式
      - L                : 线性拉伸
      - P                : Padding 
      - F                : 翻转
  - 调用batch_processor函数后将直接保存批处理的图像到指定文件夹
</pre>

**_Data_Loader_**
- **数据加载模块**
- **适应于单输入图像网络**
- **不进行任何数据预处理**
- 用于将样本进行读取
  - 读取后将栅格存储进入Dataloader中
  - 返回为栅格序列
  - 返回的栅格类型为：N·C·H·W
<pre>
- DataLoader            : 定义了一个DataLoader类
  - input_datasets_path : 输入数据集路径
- 返回一对数据数组(image, label)
Attention!:数据集路径文件夹应当按照一下定义要求进行分配：
--Datasets
 ->image1(模型输入1文件夹)
 ->image2(模型输入2文件夹)
 ...
 ->label(标签文件夹)
</pre>

### 网络库
#### 包含：
**_U-Net_** ：经典的U-Net语义分割模型

**_AG-U-Net_** ：于跳层链接&上采样间增加注意力门

**_ASPP-U-Net_** ：增加ASPP空间金字塔池化模块

**_DeepLabV3+_** ：经典的DeepLabV3+语义分割模型

**_SegFormer_** ：经典的SegFormer语义分割模型

**_SegFormer-OutConv_** ：将最后的多层感知机解码器更换为多层卷积解码器

**_BenchSegNet_** ：为面向露天矿台阶识别任务修改的SegFormer变体
#### 参数调整：
<pre>
- in_channels ：输入通道数
- num_classes ：输出类别数
- backbone    ：骨干网络
- pretrained  ：使用预训练参数
</pre>

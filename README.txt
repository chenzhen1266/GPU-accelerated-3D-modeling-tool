# This code is a conversion tool for batch conversion of NIfTI (.nii/.nii.gz) medical images into STL (.stl) 3D models.
# Input format: (.nii/.nii.gz); Output format: (.stl)

Main functions:
# 1. CUDA accelerated interpolation: trilinear interpolation calculation using PyTorch; data is processed in GPU memory for accelerated calculation; supports arbitrary scale factors (example uses 2x interpolation)
# 2. Improved mesh processing flow: use vtkMarchingCubes with intermediate threshold (0.5) to extract isosurfaces; enhanced smoothing parameter setting (50 iterations + low-pass filtering); add mesh simplification steps to optimize output quality
# 3. Data preprocessing: explicit binarization to ensure data consistency; automatic processing of data dimensions and spatial parameters
# 4. Error handling: explicit check of CUDA availability; improved array dimension handling
# 5. Modular processing and centralized control

# Methods:
# Upsampling: trilinear interpolation upsampling
# Smoothing: median filtering, enhanced smoothing and mesh simplification
# Dilation: three-dimensional dilation structure element. The shape of the structural element is an anisotropic cross, and the structural element can be expanded by adjusting the radius.
# When radius is 1, the structural element includes the center point and six adjacent points (front, back, left, right, top, and bottom). ; When radius is 2, the structural element is expanded to 26 neighborhoods, including all possible adjacent points.
# dilation_iterations: dilation iterations (controls dilation amplitude)
# scale_factor: trilinear interpolation upsampling multiple
# smooth_iterations: smoothing filter iterations
# median_filter_size: median filter kernel size
# target_reduction: mesh simplification ratio

# Parameter safety range recommendations:
# Dilation times: 0-3 times recommended (0 means no dilation)
# Upsampling multiple: 1-4 times recommended (too high will cause memory explosion)
# Median filter kernel: must be an odd number (3/5/7)
# Smoothing iterations: 50-500 times (too high will cause over-smoothing)
# Simplification rate: 0-0.9 (0.9 means retaining 10% of the patches)










# 该代码是一个 NIfTI（.nii/.nii.gz）医学影像 批量转换为 STL（.stl）三维模型 的转换工具。
# 输入格式：（.nii/.nii.gz）；输出格式：（.stl）


主要功能：
# 1.CUDA加速插值：使用PyTorch进行三线性插值计算;数据在GPU内存中进行处理加速计算;支持任意比例因子（示例使用2倍插值）
# 2.改进的网格处理流程：改用vtkMarchingCubes配合中间阈值（0.5）提取等值面;增强的平滑参数设置（50次迭代+低通滤波）;添加网格简化步骤优化输出质量
# 3.数据预处理：显式的二值化处理确保数据一致性;自动处理数据维度和空间参数
# 4.错误处理：显式检查CUDA可用性;改进的数组维度处理
# 5.模块化处理和集中控制


# 方法：
# 上采样：三线性插值上采样
# 平滑：中值滤波、增强型平滑处理和网格简化
# 膨胀：三维的膨胀结构元素。结构元素的形状是一个各向异性的十字形，可以通过调整半径来扩展结构元素。
# radius 为 1 时，结构元素包括中心点和六个相邻点（前、后、左、右、上、下）。；radius 为 2 时，结构元素扩展为 26 邻域，包括所有可能的相邻点。

# dilation_iterations: 膨胀迭代次数（控制膨胀幅度）
# scale_factor: 三线性插值上采样倍数
# smooth_iterations: 平滑滤波器迭代次数
# median_filter_size: 中值滤波核尺寸
# target_reduction: 网格简化比例

# 参数安全范围建议：
# 膨胀次数：推荐0-3次（0表示不膨胀）
# 上采样倍数：推荐1-4倍（过高会导致内存暴涨）
# 中值滤波核：必须为奇数（3/5/7）
# 平滑迭代：50-500次（过高会导致过度平滑）
# 简化率：0-0.9（0.9表示保留10%面片）

import os
import numpy as np
import vtk
from vtk.util import numpy_support
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter, binary_dilation, binary_erosion

# ---------------------------
# 核心参数配置区
# ---------------------------
control_params = {
    'label': 1,  # 分割标签ID
    'scale_factor': 2,  # 上采样倍数（1-4）
    'median_filter_size': 3,  # 中值滤波核尺寸（奇数）
    'smooth_iterations': 300,  # 平滑迭代次数（50-1000）【平滑次数多会导致类似于侵蚀的效果】
    'dilation_iterations': 1,  # 膨胀次数（0-3）
    'erosion_iterations': 1,  # 侵蚀次数（0-3）
    'morph_radius': 1,  # 结构元素半径（1-2）
    'target_reduction': 0.3  # 网格简化率（0-0.9）
}


# ---------------------------
# 图像处理函数集
# ---------------------------
def create_morph_structure(radius):
    """创建各向异性结构元素"""
    structure = np.zeros((3, 3, 3), dtype=bool)

    # 基础6邻域连接
    structure[1, 1, [0, 1, 2]] = True  # 前后中轴
    structure[1, [0, 1, 2], 1] = True  # 左右中轴
    structure[[0, 1, 2], 1, 1] = True  # 上下中轴

    # 扩展半径连接
    if radius >= 2:
        structure[0:3:2, 0:3:2, 0:3:2] = True  # 8个顶点

    return structure


def interpolate_volume(data, scale_factor):
    """CUDA加速的三线性插值"""
    with torch.no_grad():
        data_tensor = torch.from_numpy(data).float().cuda()
        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
        interpolated = F.interpolate(
            data_tensor,
            scale_factor=scale_factor,
            mode='trilinear',
            align_corners=False
        )
        return interpolated.squeeze().cpu().numpy()


def smooth_volume(data, filter_size):
    """中值滤波平滑"""
    return median_filter(data, size=filter_size)


def morph_operation(data, dilate_iter, erode_iter, radius):
    """组合形态学操作"""
    structure = create_morph_structure(radius)
    processed = data.copy()

    # 膨胀阶段
    for _ in range(dilate_iter):
        processed = binary_dilation(processed, structure=structure)

    # 侵蚀阶段
    for _ in range(erode_iter):
        processed = binary_erosion(processed, structure=structure)

    return processed.astype(np.float32)


# ---------------------------
# 主处理流程
# ---------------------------
def nii_2_mesh(filename_nii, filename_stl, params):
    """
    优化后的处理流程：
    1. 数据插值
    2. 平滑处理
    3. 形态学操作
    """
    # 硬件检查
    assert torch.cuda.is_available(), "CUDA GPU不可用"

    # 读取原始数据
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    vtk_image = reader.GetOutput()

    # 数据预处理
    np_data = numpy_support.vtk_to_numpy(
        vtk_image.GetPointData().GetScalars()
    ).reshape(vtk_image.GetDimensions(), order='F')
    np_data = (np_data == params['label']).astype(np.float32)

    # 执行处理流水线
    interpolated = interpolate_volume(np_data, params['scale_factor'])
    smoothed = smooth_volume(interpolated, params['median_filter_size'])
    morphed = morph_operation(
        smoothed,
        params['dilation_iterations'],
        params['erosion_iterations'],
        params['morph_radius']
    )

    # 构建VTK图像
    new_image = vtk.vtkImageData()
    new_image.SetDimensions(*morphed.shape)
    new_image.SetSpacing([sp / params['scale_factor'] for sp in vtk_image.GetSpacing()])
    new_image.SetOrigin(vtk_image.GetOrigin())

    vtk_array = numpy_support.numpy_to_vtk(
        morphed.ravel(order='F'),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    new_image.GetPointData().SetScalars(vtk_array)

    # 等值面提取
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(new_image)
    marching_cubes.SetValue(0, 0.5)
    marching_cubes.Update()

    # 网格优化
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(params['smooth_iterations'])
    smoother.SetPassBand(0.01)
    smoother.NonManifoldSmoothingOn()
    smoother.Update()

    # 网格简化
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(smoother.GetOutputPort())
    decimator.SetTargetReduction(params['target_reduction'])
    decimator.PreserveTopologyOn()
    decimator.Update()

    # 输出STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filename_stl)
    writer.SetInputConnection(decimator.GetOutputPort())
    writer.Write()
    print(f"生成模型: {filename_stl}")


# ---------------------------
# 批量处理
# ---------------------------
def process_nii_files(input_folder, output_folder, params):
    """批量处理NIfTI文件"""
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith((".nii", ".nii.gz")):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(
                output_folder,
                file.replace(".nii.gz", ".stl").replace(".nii", ".stl")
            )
            nii_2_mesh(input_path, output_path, params)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_nii_files(
        input_folder=os.path.join(script_dir, "input"),
        output_folder=os.path.join(script_dir, "output"),
        params=control_params
    )




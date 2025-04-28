import numpy as np
import torch
import nibabel as nib

def mni_to_voxel(mni_coord, affine=None, template_path=None):
    """
    Convert MNI coordinates to voxel coordinates
    
    Parameters:
    mni_coord: list or numpy.ndarray, MNI coordinates [x, y, z]
    affine: numpy.ndarray, affine transformation matrix for conversion
    template_path: str, path to MNI template file, used to get affine matrix if not provided
    
    Returns:
    voxel_coord: list, voxel coordinates [x, y, z]
    """
    # If no affine matrix and no template path are provided
    if affine is None and template_path is None:
        # Try to load the standard MNI152 template
        try:
            # First check in the current directory
            template_path = "../data/MNI152_T1_1mm.nii.gz"
            template_img = nib.load(template_path)
            print(f"Using MNI152 template from: {template_path}")
            affine = template_img.affine
        except (FileNotFoundError, ImportError):
            try:
                # Then try in the data directory
                template_path = "data/MNI152_T1_1mm.nii.gz"
                template_img = nib.load(template_path)
                print(f"Using MNI152 template from: {template_path}")
                affine = template_img.affine
            except (FileNotFoundError, ImportError):
                # If template still not found, use default approximate affine matrix
                print("Warning: MNI152 template not found. Using default approximate affine matrix.")
                affine = np.array([
                    [-1, 0, 0, 90],
                    [0, 1, 0, -126],
                    [0, 0, 1, -72],
                    [0, 0, 0, 1]
                ])
    elif affine is None:
        # Load affine matrix from template file
        try:
            template_img = nib.load(template_path)
            affine = template_img.affine
            print(f"Using template from: {template_path}")
        except (FileNotFoundError, ImportError) as e:
            print(f"Error loading template: {e}")
            print("Using default approximate affine matrix.")
            affine = np.array([
                [-1, 0, 0, 90],
                [0, 1, 0, -126],
                [0, 0, 1, -72],
                [0, 0, 0, 1]
            ])
    
    # Convert MNI coordinate to homogeneous coordinate
    if isinstance(mni_coord, list):
        mni_coord = np.array(mni_coord)
    
    mni_homo = np.append(mni_coord, 1)
    
    # Calculate voxel coordinate using affine inverse
    affine_inv = np.linalg.inv(affine)
    voxel_homo = affine_inv @ mni_homo
    voxel_coord = voxel_homo[:3]
    
    # Round to integer coordinates
    voxel_coord = np.round(voxel_coord).astype(int)
    
    return voxel_coord.tolist()

def voxel_to_mni(voxel_coord, affine=None, template_path=None):
    """
    Convert voxel coordinates to MNI coordinates
    
    Parameters:
    voxel_coord: list or numpy.ndarray, voxel coordinates [x, y, z]
    affine: numpy.ndarray, affine transformation matrix for conversion
    template_path: str, path to MNI template file, used to get affine matrix if not provided
    
    Returns:
    mni_coord: list, MNI coordinates [x, y, z]
    """
    # If no affine matrix and no template path are provided
    if affine is None and template_path is None:
        # Try to load the standard MNI152 template
        try:
            # First check in the current directory
            template_path = "../data/MNI152_T1_1mm.nii.gz"
            template_img = nib.load(template_path)
            print(f"Using MNI152 template from: {template_path}")
            affine = template_img.affine
        except (FileNotFoundError, ImportError):
            try:
                # Then try in the data directory
                template_path = "data/MNI152_T1_1mm.nii.gz"
                template_img = nib.load(template_path)
                print(f"Using MNI152 template from: {template_path}")
                affine = template_img.affine
            except (FileNotFoundError, ImportError):
                # If template still not found, use default approximate affine matrix
                print("Warning: MNI152 template not found. Using default approximate affine matrix.")
                affine = np.array([
                    [-1, 0, 0, 90],
                    [0, 1, 0, -126],
                    [0, 0, 1, -72],
                    [0, 0, 0, 1]
                ])
    elif affine is None:
        # Load affine matrix from template file
        try:
            template_img = nib.load(template_path)
            affine = template_img.affine
            print(f"Using template from: {template_path}")
        except (FileNotFoundError, ImportError) as e:
            print(f"Error loading template: {e}")
            print("Using default approximate affine matrix.")
            affine = np.array([
                [-1, 0, 0, 90],
                [0, 1, 0, -126],
                [0, 0, 1, -72],
                [0, 0, 0, 1]
            ])
    
    # Convert voxel coordinate to homogeneous coordinate
    if isinstance(voxel_coord, list):
        voxel_coord = np.array(voxel_coord)
    
    voxel_homo = np.append(voxel_coord, 1)
    
    # Calculate MNI coordinate
    mni_homo = affine @ voxel_homo
    mni_coord = mni_homo[:3]
    
    return mni_coord.tolist()

def create_roi_mask(shape, coordinates, roi_size=3):
    """
    根据坐标创建ROI掩码
    
    参数:
    shape: tuple，输出掩码的形状
    coordinates: list，体素坐标列表，每个坐标是[x, y, z]格式
    roi_size: int 或 list，ROI的大小，如果为int，则表示立方体边长；如果为list，则表示[x_size, y_size, z_size]
    
    返回:
    mask: torch.Tensor，ROI掩码，形状为shape
    """
    # 创建掩码
    mask = torch.zeros(shape)
    
    # 处理roi_size
    if isinstance(roi_size, int):
        roi_size = [roi_size, roi_size, roi_size]
    
    # 计算半径
    half_rx = roi_size[0] // 2
    half_ry = roi_size[1] // 2
    half_rz = roi_size[2] // 2
    
    # 遍历每个坐标
    for coord in coordinates:
        x, y, z = coord
        
        # 计算ROI范围
        x_start = max(0, x - half_rx)
        x_end = min(shape[0], x + half_rx + 1)
        y_start = max(0, y - half_ry)
        y_end = min(shape[1], y + half_ry + 1)
        z_start = max(0, z - half_rz)
        z_end = min(shape[2], z + half_rz + 1)
        
        # 设置ROI区域为1
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = 1
    
    return mask

def find_peak_coordinates(stat_map, n_peaks=10, min_distance=20):
    """
    在统计图中查找峰值坐标
    
    参数:
    stat_map: torch.Tensor，统计图
    n_peaks: int，要查找的峰值数量
    min_distance: int，峰值之间的最小距离（以体素为单位）
    
    返回:
    peak_coords: list，峰值坐标列表，每个坐标是[x, y, z]格式
    peak_values: list，峰值值列表
    """
    # 转换为numpy数组以便使用scipy
    if isinstance(stat_map, torch.Tensor):
        stat_map_np = stat_map.numpy()
    else:
        stat_map_np = stat_map
    
    # 找到局部最大值
    # 注意：这里使用简化方法，实际应该使用scipy.ndimage.maximum_filter
    peak_coords = []
    peak_values = []
    
    # 复制一份数据用于标记已找到的峰值
    temp_map = stat_map_np.copy()
    
    # 迭代找峰值
    for _ in range(n_peaks):
        # 找到当前最大值
        max_idx = np.argmax(temp_map)
        max_idx = np.unravel_index(max_idx, temp_map.shape)
        max_val = temp_map[max_idx]
        
        # 如果最大值为零，则停止
        if max_val == 0:
            break
            
        peak_coords.append(list(max_idx))
        peak_values.append(float(max_val))
        
        # 清除以该峰值为中心的区域，避免重复选择
        x, y, z = max_idx
        x_start = max(0, x - min_distance // 2)
        x_end = min(temp_map.shape[0], x + min_distance // 2 + 1)
        y_start = max(0, y - min_distance // 2)
        y_end = min(temp_map.shape[1], y + min_distance // 2 + 1)
        z_start = max(0, z - min_distance // 2)
        z_end = min(temp_map.shape[2], z + min_distance // 2 + 1)
        
        temp_map[x_start:x_end, y_start:y_end, z_start:z_end] = 0
    
    return peak_coords, peak_values
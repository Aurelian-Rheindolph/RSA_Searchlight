import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def load_nifti_data(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def compute_rdm(data):
    """
    计算全脑整体RDM（不设窗口），输入data: (n_conditions, x, y, z)，返回n_conditions x n_conditions的RDM。
    只选取所有条件下都非nan的体素进行RDM计算。
    使用torch加速，dissimilarity为1-皮尔逊相关系数。
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_conditions = data.shape[0]
    data_2d = data.reshape(n_conditions, -1)
    
    # 在NumPy中处理NaN值，因为PyTorch的isnan处理更复杂
    valid_mask = ~np.isnan(data_2d).any(axis=0)
    if valid_mask.sum() == 0:
        raise ValueError("所有体素均为nan，无法计算RDM！")
    data_valid = data_2d[:, valid_mask]
    
    # 将数据移至GPU
    data_t = torch.from_numpy(data_valid).float().to(device)
    
    # 一次性计算相关系数矩阵
    corr = torch.corrcoef(data_t)
    corr = torch.nan_to_num(corr, nan=0.0)
    rdm = 1 - corr
    
    # 在GPU上进行标准化 - 使用mask_select替代nanmin/nanmax
    valid_mask = ~torch.isnan(rdm)
    if valid_mask.any():
        # 只考虑非NaN值来计算最小值和最大值
        valid_values = rdm.masked_select(valid_mask)
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val > min_val:
            rdm = (rdm - min_val) / (max_val - min_val)
        else:
            rdm = torch.zeros_like(rdm)
    else:
        rdm = torch.zeros_like(rdm)
    
    # 将结果移回CPU
    rdm_np = rdm.cpu().numpy()
    return rdm_np

def searchlight_rdm(beta_files, mask_data=None, window_size=(3, 3, 3), batch_size=128):
    n_conditions = len(beta_files)
    betas = []
    for beta_path in beta_files:
        beta, _ = load_nifti_data(beta_path)
        betas.append(beta)
    betas = np.stack(betas, axis=0)  # (n_conditions, x, y, z)
    
    shape = betas.shape[1:]
    wx, wy, wz = window_size
    rx, ry, rz = wx//2, wy//2, wz//2
    
    # 创建输出RDM数组
    rdm_map = np.zeros((n_conditions, n_conditions, *shape), dtype=np.float32)
    
    # 移动数据到GPU (如果可用)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    betas_t = torch.from_numpy(betas).float().to(device)
    # 将掩码也移至GPU以加速计算
    mask_t = None
    if mask_data is not None:
        mask_t = torch.from_numpy(mask_data).bool().to(device)
    
    # 准备处理的坐标
    coords = []
    for x in range(rx, shape[0]-rx):
        for y in range(ry, shape[1]-ry):
            for z in range(rz, shape[2]-rz):
                if mask_t is not None and not mask_t[x, y, z]:
                    continue
                coords.append((x, y, z))
    
    # 创建GPU上的RDM映射
    rdm_map_t = torch.zeros((n_conditions, n_conditions, *shape), dtype=torch.float32, device=device)
    
    # 分批处理坐标
    for batch_idx in tqdm(range(0, len(coords), batch_size), desc="Processing voxels"):
        batch_coords = coords[batch_idx:batch_idx+batch_size]
        
        # 批量提取每个体素的窗口数据
        batch_patches = []
        for x, y, z in batch_coords:
            patch = betas_t[:, x-rx:x+rx+1, y-ry:y+ry+1, z-rz:z+rz+1].reshape(n_conditions, -1)
            valid = ~torch.isnan(patch).any(dim=0)
            if valid.sum() < 5:
                # 如果有效体素太少，填充为0
                patch_valid = torch.zeros((n_conditions, 5), device=device)
            else:
                patch_valid = patch[:, valid]
            batch_patches.append(patch_valid)
        
        # 对每对条件计算相关性
        for cond_i in range(n_conditions):
            for cond_j in range(cond_i+1, n_conditions):
                for idx, (x, y, z) in enumerate(batch_coords):
                    patch_valid = batch_patches[idx]
                    if patch_valid.shape[1] < 5:
                        rdm_map_t[cond_i, cond_j, x, y, z] = rdm_map_t[cond_j, cond_i, x, y, z] = 0
                        continue
                    
                    # 计算两个条件之间的相关性
                    data_i = patch_valid[cond_i]
                    data_j = patch_valid[cond_j]
                    
                    # 中心化数据
                    data_i = data_i - torch.mean(data_i)
                    data_j = data_j - torch.mean(data_j)
                    
                    # 计算相关系数
                    corr = torch.sum(data_i * data_j) / (torch.sqrt(torch.sum(data_i**2) * torch.sum(data_j**2)) + 1e-8)
                    
                    # 计算RDM (1-相关性)
                    rdm_val = 1 - corr
                    rdm_map_t[cond_i, cond_j, x, y, z] = rdm_map_t[cond_j, cond_i, x, y, z] = rdm_val
    
    # 在GPU上标准化RDM到0-1范围 - 使用mask_select替代nanmin/nanmax
    valid_mask = ~torch.isnan(rdm_map_t)
    if valid_mask.any():
        valid_values = rdm_map_t.masked_select(valid_mask)
        rdm_map_min = valid_values.min()
        rdm_map_max = valid_values.max()
        
        if rdm_map_max > rdm_map_min:
            rdm_map_t = (rdm_map_t - rdm_map_min) / (rdm_map_max - rdm_map_min)
    else:
        # 如果都是NaN，则将其全部设为0
        rdm_map_t = torch.zeros_like(rdm_map_t)
    
    # 将结果移回CPU
    rdm_map = rdm_map_t.cpu().numpy()
    return rdm_map

def mni_to_voxel(mni_coords, affine):
    inv_affine = np.linalg.inv(affine)
    mni_homogeneous = np.append(mni_coords, 1)
    voxel_homogeneous = inv_affine.dot(mni_homogeneous)
    voxel_coords = np.round(voxel_homogeneous[:3]).astype(int)
    return tuple(voxel_coords)

def extract_roi_window(data, center, size=3):
    """提取指定中心点周围的数据窗口，使用PyTorch加速"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_t = torch.from_numpy(data).float().to(device) if isinstance(data, np.ndarray) else data
    
    x, y, z = center
    r = size // 2
    
    # 提取窗口数据并重塑
    window_data = data_t[:, x-r:x+r+1, y-r:y+r+1, z-r:z+r+1].reshape(data_t.shape[0], -1)
    
    # 如果输入是numpy数组，则返回numpy数组以保持兼容性
    if isinstance(data, np.ndarray):
        return window_data.cpu().numpy()
    return window_data

def zscore_betas(betas):
    """对每个beta图做z-score标准化，去除无关信号，使用PyTorch加速"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    betas_t = torch.from_numpy(betas).float().to(device)
    
    # 计算每个条件下的均值和标准差
    # 保持维度以便进行广播
    # PyTorch没有nanmean和nanstd，使用手动mask替代
    mask = ~torch.isnan(betas_t)
    
    # 计算均值，处理NaN
    sum_values = torch.where(mask, betas_t, torch.zeros_like(betas_t)).sum(dim=(1,2,3), keepdim=True)
    count = mask.sum(dim=(1,2,3), keepdim=True).clamp(min=1)  # 防止除0
    mean = sum_values / count
    
    # 计算标准差，处理NaN
    squared_diff = torch.where(mask, (betas_t - mean)**2, torch.zeros_like(betas_t))
    variance = squared_diff.sum(dim=(1,2,3), keepdim=True) / count.clamp(min=1)
    std = torch.sqrt(variance) + 1e-8
    
    # 进行归一化
    betas_norm = torch.where(mask, (betas_t - mean) / std, torch.zeros_like(betas_t))
    
    # 返回NumPy数组以便与其他函数兼容
    return betas_norm.cpu().numpy()

def normalize_rdm(rdm):
    """将RDM标准化到0-1区间，使用PyTorch加速"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rdm_t = torch.from_numpy(rdm).float().to(device)
    
    # 使用mask_select替代nanmin/nanmax
    valid_mask = ~torch.isnan(rdm_t)
    if valid_mask.any():
        valid_values = rdm_t.masked_select(valid_mask)
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val > min_val:
            rdm_t = torch.where(valid_mask, (rdm_t - min_val) / (max_val - min_val), torch.zeros_like(rdm_t))
        else:
            rdm_t = torch.zeros_like(rdm_t)
    else:
        rdm_t = torch.zeros_like(rdm_t)
    
    return rdm_t.cpu().numpy()

def main():
    # List to store RDM results for all subjects
    all_subjects_rdm_results = []
    
    # Loop through the 30 subject folders
    for i in range(1, 31):
        subject_id = f"sub_{i:02d}"  # Format as sub_01, sub_02, etc.
        subject_dir = f"data/firstlevel_forRSA/{subject_id}"
        
        # Define paths to beta files for this subject
        beta_files = [
            f'{subject_dir}/beta_0002.nii',
            f'{subject_dir}/beta_0004.nii',
            f'{subject_dir}/beta_0006.nii'
        ]
        mask_file = f'{subject_dir}/mask.nii'
        try:
            # Load mask data
            mask_data, _ = load_nifti_data(mask_file)
            # Compute RDM for this subject
            subject_rdm_results = searchlight_rdm(beta_files, mask_data)
            # Add to the list of all subjects
            all_subjects_rdm_results.append(subject_rdm_results)
            print(f"Processed {subject_id} successfully")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    # Average RDM results across all subjects using PyTorch
    if all_subjects_rdm_results:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 将所有结果转换为张量并移至GPU
        all_subjects_tensor = torch.from_numpy(np.stack(all_subjects_rdm_results, axis=0)).float().to(device)
        # 在GPU上计算平均值
        combined_rdm_results_tensor = torch.mean(all_subjects_tensor, dim=0)
        # 移回CPU
        combined_rdm_results = combined_rdm_results_tensor.cpu().numpy()
        print(f"Successfully combined RDM results from {len(all_subjects_rdm_results)} subjects")
        return combined_rdm_results
    else:
        print("No results to combine")
        return None
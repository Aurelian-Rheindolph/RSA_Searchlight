import torch
import os
import numpy as np
import nibabel as nib

def load_data(data_dir, conditions):
    all_data = {}
    loaded_subjects = 0
    total_subjects = 30  # 默认30个被试
    subjects_missing = []
    subjects_incomplete = []
    
    # 数据维度检查
    data_shapes = {}
    
    print(f"Attempting to load data from: {os.path.abspath(data_dir)}")
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' does not exist!")
        return {}
    
    # 遍历所有被试
    for i in range(1, total_subjects + 1):
        sub_id = f"sub_{i:02d}"
        sub_path = os.path.join(data_dir, sub_id)
        
        if not os.path.exists(sub_path):
            print(f"Warning: {sub_path} does not exist, skipping")
            subjects_missing.append(sub_id)
            continue
            
        all_data[sub_id] = {}
        condition_loaded = 0
        
        # 加载每个条件的数据
        for cond_name, file_name in conditions.items():
            file_path = os.path.join(sub_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist, skipping")
                continue
                
            # 加载nii文件
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # 检查数据有效性
                if np.isnan(data).any():
                    print(f"Warning: NaN values found in {file_path}")
                if np.isinf(data).any():
                    print(f"Warning: Infinite values found in {file_path}")
                
                # 记录数据形状
                shape = data.shape
                if cond_name not in data_shapes:
                    data_shapes[cond_name] = {}
                if shape not in data_shapes[cond_name]:
                    data_shapes[cond_name][shape] = 0
                data_shapes[cond_name][shape] += 1
                
                # 存储数据和仿射变换矩阵
                all_data[sub_id][cond_name] = {
                    "data": torch.tensor(data, dtype=torch.float32),
                    "affine": img.affine
                }
                condition_loaded += 1
                
                # 打印简要数据统计信息
                print(f"Loaded {cond_name} for {sub_id}: shape={shape}, min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # 检查是否所有条件都已加载
        if condition_loaded == len(conditions):
            loaded_subjects += 1
        else:
            subjects_incomplete.append(sub_id)
            print(f"Warning: Subject {sub_id} has incomplete condition data ({condition_loaded}/{len(conditions)} conditions loaded)")
    
    # 报告数据形状一致性
    print("\nData shape consistency check:")
    for cond, shapes in data_shapes.items():
        print(f"Condition {cond}:")
        for shape, count in shapes.items():
            print(f"  Shape {shape}: {count} subjects")
    
    # 报告数据加载情况
    print(f"\nData loading summary:")
    print(f"- Subjects fully loaded: {loaded_subjects}/{total_subjects} ({loaded_subjects/total_subjects*100:.1f}%)")
    print(f"- Subjects missing: {len(subjects_missing)}")
    print(f"- Subjects with incomplete data: {len(subjects_incomplete)}")
    
    if len(subjects_missing) > 0:
        print(f"  Missing subjects: {', '.join(subjects_missing)}")
    if len(subjects_incomplete) > 0:
        print(f"  Incomplete subjects: {', '.join(subjects_incomplete)}")
    
    # 数据验证摘要
    if loaded_subjects == 0:
        print("\nCRITICAL ERROR: No subjects were successfully loaded!")
        return {}
    
    # 检查是否有足够的被试用于分析
    if loaded_subjects < total_subjects * 0.8:  # 80%阈值
        print("\nWARNING: Less than 80% of subjects were loaded completely.")
        proceed = input("Do you want to proceed with the analysis anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Analysis aborted by user.")
            exit()
    
    # 抽样打印一个被试的完整信息
    if all_data:
        sample_subject = next(iter(all_data))
        print(f"\nSample data for {sample_subject}:")
        for cond, data in all_data[sample_subject].items():
            tensor_info = data["data"]
            print(f"  {cond}: shape={tensor_info.shape}, dtype={tensor_info.dtype}")
    
    return all_data

def compute_rdm(data, method='correlation'):
    """
    计算 RDM，支持多种距离度量
    
    参数:
    data - 形状为 [条件数, 体素数] 的张量或numpy数组
    method - 距离度量方法: 'correlation', 'euclidean', 'cosine'
    
    返回:
    RDM 矩阵
    """
    # 检查输入类型，如果是NumPy数组则转换为PyTorch张量
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # 将输入张量展平为 2D 张量 [n_samples, n_features]
    n_samples = data.shape[0]
    data_flat = data.reshape(n_samples, -1)
    
    # 检查数据是否包含NaN或Inf
    if torch.isnan(data_flat).any() or torch.isinf(data_flat).any():
        print(f"警告: 输入数据包含NaN或Inf值，将被替换为0")
        data_flat = torch.nan_to_num(data_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 检查数据的标准差
    for i in range(n_samples):
        std_i = torch.std(data_flat[i])
        if std_i < 1e-6:
            print(f"警告: 第{i+1}个条件的标准差很小 ({std_i.item():.2e})，可能导致RDM计算不稳定")
    
    # 初始化RDM矩阵
    rdm = torch.zeros((n_samples, n_samples), device=data_flat.device)
    
    if method == 'correlation':
        # 使用更稳定的方式计算RDM
        for i in range(n_samples):
            rdm[i, i] = 0.0  # 对角线元素设为0
            
            for j in range(i+1, n_samples):  # 只计算上三角部分
                # 提取数据
                x = data_flat[i]
                y = data_flat[j]
                
                # 移除两个向量共同的NaN位置
                valid_mask = ~(torch.isnan(x) | torch.isnan(y))
                if not torch.any(valid_mask):
                    print(f"警告: 条件{i+1}和{j+1}之间没有共同的有效数据点")
                    rdm[i, j] = rdm[j, i] = 1.0  # 设为最大相异性
                    continue
                
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                # 计算均值
                mean_x = torch.mean(x_valid)
                mean_y = torch.mean(y_valid)
                
                # 计算协方差和标准差
                cov = torch.mean((x_valid - mean_x) * (y_valid - mean_y))
                std_x = torch.std(x_valid, unbiased=False)
                std_y = torch.std(y_valid, unbiased=False)
                
                # 计算相关系数
                if std_x < 1e-10 or std_y < 1e-10:
                    print(f"警告: 条件{i+1}或{j+1}的标准差接近于0，设置相关系数为0")
                    corr = 0.0
                else:
                    corr = cov / (std_x * std_y)
                    
                    # 确保相关系数在[-1, 1]范围内
                    if corr > 1.0:
                        print(f"修正: 相关系数大于1.0 ({corr:.4f})，设置为1.0")
                        corr = 1.0
                    elif corr < -1.0:
                        print(f"修正: 相关系数小于-1.0 ({corr:.4f})，设置为-1.0")
                        corr = -1.0
                
                # 相异度 = 1 - 相关性
                dissimilarity = 1.0 - corr.abs()
                
                # 填充RDM矩阵的上下三角
                rdm[i, j] = rdm[j, i] = dissimilarity
                
    elif method == 'euclidean':
        # 使用欧氏距离计算RDM
        for i in range(n_samples):
            rdm[i, i] = 0.0  # 对角线元素设为0
            
            for j in range(i+1, n_samples):  # 只计算上三角部分
                # 计算欧氏距离
                diff = data_flat[i] - data_flat[j]
                
                # 移除NaN
                valid_diff = diff[~torch.isnan(diff)]
                if valid_diff.numel() == 0:
                    print(f"警告: 条件{i+1}和{j+1}之间没有共同的有效数据点")
                    rdm[i, j] = rdm[j, i] = 1.0  # 设为最大相异性
                    continue
                    
                euclidean_dist = torch.sqrt(torch.sum(valid_diff * valid_diff))
                
                # 归一化距离值
                n_features = valid_diff.numel()
                normalized_dist = euclidean_dist / torch.sqrt(torch.tensor(n_features, dtype=torch.float32))
                
                # 缩放到[0,1]范围
                rdm[i, j] = rdm[j, i] = min(normalized_dist.item(), 1.0)
                
    elif method == 'cosine':
        # 使用余弦距离计算RDM
        for i in range(n_samples):
            rdm[i, i] = 0.0  # 对角线元素设为0
            
            for j in range(i+1, n_samples):  # 只计算上三角部分
                x = data_flat[i]
                y = data_flat[j]
                
                # 移除两个向量共同的NaN位置
                valid_mask = ~(torch.isnan(x) | torch.isnan(y))
                if not torch.any(valid_mask):
                    print(f"警告: 条件{i+1}和{j+1}之间没有共同的有效数据点")
                    rdm[i, j] = rdm[j, i] = 1.0  # 设为最大相异性
                    continue
                
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                # 计算余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(
                    x_valid.unsqueeze(0), 
                    y_valid.unsqueeze(0)
                )
                
                # 余弦距离 = 1 - 余弦相似度
                cos_dist = 1.0 - cos_sim
                
                # 确保在[0,1]范围内
                cos_dist = torch.clamp(cos_dist, 0.0, 1.0)
                
                rdm[i, j] = rdm[j, i] = cos_dist.item()
    else:
        raise ValueError(f"不支持的距离度量方法: {method}，支持的方法有: correlation, euclidean, cosine")
    
    # 打印RDM的基本统计信息
    non_diag_mask = ~torch.eye(n_samples, dtype=torch.bool, device=rdm.device)
    non_diag_values = rdm[non_diag_mask]
    
    print(f"RDM统计信息 - 方法: {method}")
    print(f"  形状: {rdm.shape}")
    print(f"  非对角元素: 最小值={non_diag_values.min().item():.4f}, 最大值={non_diag_values.max().item():.4f}, 平均值={non_diag_values.mean().item():.4f}")
    print(f"  NaN或Inf值: {torch.isnan(rdm).any().item() or torch.isinf(rdm).any().item()}")
    
    # 确保没有NaN或Inf值
    if torch.isnan(rdm).any() or torch.isinf(rdm).any():
        print("警告: RDM中包含NaN或Inf值，将被替换为1.0")
        rdm = torch.nan_to_num(rdm, nan=1.0, posinf=1.0, neginf=1.0)
    
    return rdm

def create_brain_map_from_rdms(rdm_volume, method='mean'):
    # 将torch张量转换为numpy数组
    if isinstance(rdm_volume, torch.Tensor):
        rdm_volume_np = rdm_volume.detach().cpu().numpy()
    else:
        rdm_volume_np = rdm_volume
    
    # 提取RDM体积最后两个维度以外的形状
    spatial_shape = rdm_volume_np.shape[:-2]
    
    # 创建脑激活图
    brain_map = np.zeros(spatial_shape)
    
    # 对每个体素的RDM进行统计
    if method == 'mean':
        for i in range(spatial_shape[0]):
            for j in range(spatial_shape[1]):
                for k in range(spatial_shape[2]):
                    rdm = rdm_volume_np[i, j, k]
                    # 计算非对角线元素的平均值
                    mask = ~np.eye(rdm.shape[0], dtype=bool)
                    brain_map[i, j, k] = np.mean(rdm[mask])
    
    elif method == 'max':
        for i in range(spatial_shape[0]):
            for j in range(spatial_shape[1]):
                for k in range(spatial_shape[2]):
                    rdm = rdm_volume_np[i, j, k]
                    # 计算非对角线元素的最大值
                    mask = ~np.eye(rdm.shape[0], dtype=bool)
                    brain_map[i, j, k] = np.max(rdm[mask])
    
    return brain_map

def save_rdm_volume(rdm_volume, affine, output_path):
    # 转换为numpy数组
    if isinstance(rdm_volume, torch.Tensor):
        rdm_np = rdm_volume.detach().cpu().numpy()
    else:
        rdm_np = rdm_volume
    
    # 创建一个简化的指标作为输出
    n_conditions = rdm_np.shape[-1]
    mask = ~np.eye(n_conditions, dtype=bool)
    mean_rdm = np.zeros(rdm_np.shape[:-2])
    for i in range(rdm_np.shape[0]):
        for j in range(rdm_np.shape[1]):
            for k in range(rdm_np.shape[2]):
                mean_rdm[i, j, k] = np.mean(rdm_np[i, j, k][mask])
    
    # 创建NIfTI对象并保存
    nii_img = nib.Nifti1Image(mean_rdm, affine)
    nib.save(nii_img, output_path)
    print(f"RDM volume saved to {output_path}")
    return mean_rdm

def masked_pairwise_rdm(all_data, condition1, condition2, mask):
    cond1_data = []
    cond2_data = []
    for sub_id, sub_data in all_data.items():
        if condition1 in sub_data and condition2 in sub_data:
            cond1_data.append(sub_data[condition1]["data"])
            cond2_data.append(sub_data[condition2]["data"])
    if not cond1_data or not cond2_data:
        raise ValueError("No data for pairwise RDM!")
    cond1_tensor = torch.stack(cond1_data)
    cond2_tensor = torch.stack(cond2_data)
    cond1_mean = torch.mean(cond1_tensor, dim=0)
    cond2_mean = torch.mean(cond2_tensor, dim=0)
    # 只保留mask内体素且两条件都非nan
    mask_idx = (mask > 0)
    valid_idx = mask_idx & (~torch.isnan(cond1_mean)) & (~torch.isnan(cond2_mean))
    print(f"pairwise mask sum: {mask_idx.sum()}, valid_idx sum: {valid_idx.sum()}")
    cond1_masked = cond1_mean[valid_idx]
    cond2_masked = cond2_mean[valid_idx]
    print(f"cond1_masked shape: {cond1_masked.shape}, mean: {cond1_masked.mean().item()}, std: {cond1_masked.std().item()}")
    print(f"cond2_masked shape: {cond2_masked.shape}, mean: {cond2_masked.mean().item()}, std: {cond2_masked.std().item()}")
    if cond1_masked.numel() == 0 or cond2_masked.numel() == 0:
        raise ValueError("No valid voxels for pairwise RDM!")
    combined = torch.stack([cond1_masked, cond2_masked])
    return compute_rdm(combined)

def load_and_combine_masks(data_dir, subject_ids, mask_filename='mask.nii'):
    """
    加载所有被试的mask并取并集，返回numpy数组（1为mask内，0为mask外）
    """
    combined_mask = None
    for sub_id in subject_ids:
        mask_path = os.path.join(data_dir, sub_id, mask_filename)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {sub_id}, skipping: {mask_path}")
            continue
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_bin = (mask_data > 0).astype(np.uint8)
        if combined_mask is None:
            combined_mask = mask_bin
        else:
            combined_mask = np.logical_or(combined_mask, mask_bin)
    if combined_mask is None:
        raise RuntimeError("No masks found!")
    return combined_mask.astype(np.uint8)


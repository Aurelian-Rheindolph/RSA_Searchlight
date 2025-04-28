import torch
import numpy as np
from tqdm import tqdm

from rdm_utils import compute_rdm

def searchlight_analysis(all_data, window_size=[3, 3, 3], method="correlation"):
    # 检查被试数据和条件的完整性
    total_subjects = len(all_data)
    condition_counts = {}
    complete_subjects = []
    
    # 确定所有可能的条件
    all_conditions = set()
    for sub_data in all_data.values():
        all_conditions.update(sub_data.keys())
    
    # 计算每个条件的被试数量
    for condition in all_conditions:
        condition_counts[condition] = sum(1 for sub_data in all_data.values() if condition in sub_data)
    
    # 找出拥有所有条件的被试
    for sub_id, sub_data in all_data.items():
        if len(sub_data) == len(all_conditions):
            complete_subjects.append(sub_id)
    
    # 报告被试和条件完整性
    print(f"\nSearchlight analysis subject data summary:")
    print(f"- Total subjects: {total_subjects}")
    print(f"- Subjects with complete data: {len(complete_subjects)} ({len(complete_subjects)/total_subjects*100:.1f}%)")
    print("- Condition availability:")
    
    for condition, count in condition_counts.items():
        print(f"  * {condition}: {count}/{total_subjects} subjects ({count/total_subjects*100:.1f}%)")
    
    # 选择第一个完整被试数据获取基本维度信息
    first_subj = complete_subjects[0] if complete_subjects else list(all_data.keys())[0]
    first_cond = list(all_data[first_subj].keys())[0]
    data_shape = all_data[first_subj][first_cond]["data"].shape
    
    print(f"- Using subject {first_subj} for reference dimensions: {data_shape}")
    
    # 创建存储每个位置RDM的结果矩阵
    # 每个位置保存RDM矩阵，RDM的大小取决于条件数量
    n_conditions = len(all_conditions)
    rdm_shape = (n_conditions, n_conditions)
    
    # 初始化结果矩阵，排除窗口边缘
    pad_x, pad_y, pad_z = window_size[0] // 2, window_size[1] // 2, window_size[2] // 2
    result_shape = (
        data_shape[0] - 2 * pad_x,
        data_shape[1] - 2 * pad_y,
        data_shape[2] - 2 * pad_z,
        rdm_shape[0],
        rdm_shape[1]
    )
    
    # 创建结果张量
    rdm_results = torch.zeros(result_shape)
    
    print(f"- Searchlight window size: {window_size}")
    print(f"- Result shape: {result_shape}")
    
    # 处理每个被试的数据并计算RDM
    subjects_data = {}
    for sub_id, sub_data in all_data.items():
        subjects_data[sub_id] = {}
        for cond_name, cond_data in sub_data.items():
            subjects_data[sub_id][cond_name] = cond_data["data"]
    
    # 计算有效体素数量以显示进度
    total_voxels = (data_shape[0] - 2 * pad_x) * (data_shape[1] - 2 * pad_y) * (data_shape[2] - 2 * pad_z)
    processed_voxels = 0
    valid_voxels = 0
    empty_voxels = 0
    
    # 遍历体积中的每个位置
    print(f"\nStarting searchlight analysis across {total_voxels} voxels...")
    
    progress_bar = tqdm(total=total_voxels)
    
    # 遍历体积中的每个位置
    for x in range(pad_x, data_shape[0] - pad_x):
        for y in range(pad_y, data_shape[1] - pad_y):
            for z in range(pad_z, data_shape[2] - pad_z):
                # 提取当前窗口内的数据
                window_data = extract_window_data(subjects_data, x, y, z, window_size)
                
                # 如果窗口内没有足够的有效数据，跳过
                if window_data is None:
                    empty_voxels += 1
                else:
                    # 计算当前位置的RDM
                    local_rdm = compute_rdm(window_data, method=method)
                    
                    # 将RDM存储到结果矩阵中
                    rdm_results[x - pad_x, y - pad_y, z - pad_z] = local_rdm
                    valid_voxels += 1
                
                processed_voxels += 1
                progress_bar.update(1)
    
    progress_bar.close()
    
    # 报告处理结果
    print(f"\nSearchlight analysis completed:")
    print(f"- Processed voxels: {processed_voxels}/{total_voxels}")
    print(f"- Valid voxels with RDM: {valid_voxels} ({valid_voxels/total_voxels*100:.1f}%)")
    print(f"- Empty/skipped voxels: {empty_voxels} ({empty_voxels/total_voxels*100:.1f}%)")
    
    return rdm_results

def extract_window_data(subjects_data, center_x, center_y, center_z, window_size):
    """
    提取以指定位置为中心的窗口数据，并按条件整合所有被试
    """
    half_wx = window_size[0] // 2
    half_wy = window_size[1] // 2
    half_wz = window_size[2] // 2
    
    # 获取数据形状（假设所有被试数据形状一致）
    first_sub = next(iter(subjects_data))
    first_cond = next(iter(subjects_data[first_sub]))
    data_shape = subjects_data[first_sub][first_cond].shape
    
    # 计算窗口范围（确保在数据范围内）
    x_start = max(0, center_x - half_wx)
    x_end = min(data_shape[0], center_x + half_wx + 1)
    y_start = max(0, center_y - half_wy)
    y_end = min(data_shape[1], center_y + half_wy + 1)
    z_start = max(0, center_z - half_wz)
    z_end = min(data_shape[2], center_z + half_wz + 1)
    
    # 检查窗口大小是否有效
    if (x_end - x_start) < 2 or (y_end - y_start) < 2 or (z_end - z_start) < 2:
        # 窗口太小，跳过
        return None
    
    # 获取所有可能的条件
    all_conditions = set()
    for sub_data in subjects_data.values():
        all_conditions.update(sub_data.keys())
    
    # 按条件收集数据
    condition_data = {}
    
    # 调试信息（每1000个体素打印一次）
    debug_flag = (center_x * 10000 + center_y * 100 + center_z) % 1000 == 0
    
    # 遍历每个条件
    for cond_name in sorted(all_conditions):
        # 初始化条件数据列表
        condition_data[cond_name] = []
        
        # 收集所有被试在当前条件下的窗口数据
        for sub_id, sub_data in subjects_data.items():
            if cond_name in sub_data:
                try:
                    # 提取当前被试当前条件的窗口数据
                    data_tensor = sub_data[cond_name]  # 直接访问数据，无需["data"]
                    window = data_tensor[x_start:x_end, y_start:y_end, z_start:z_end]
                    
                    # 检查窗口是否为空
                    if window.numel() == 0:
                        if debug_flag:
                            print(f"Empty window at ({center_x},{center_y},{center_z})")
                        continue
                    
                    # 检查窗口数据是否有效
                    if torch.isnan(window).any() or torch.isinf(window).any():
                        if debug_flag:
                            print(f"NaN/Inf values in window at ({center_x},{center_y},{center_z})")
                        continue
                    
                    # 标准差检查 - 确保有足够的变化
                    flat_window = window.flatten()
                    if torch.std(flat_window) < 1e-6:
                        if debug_flag:
                            print(f"Low variance in window at ({center_x},{center_y},{center_z})")
                        continue
                        
                    # 将3D窗口展平为1D向量
                    condition_data[cond_name].append(flat_window)
                    
                except Exception as e:
                    if debug_flag:
                        print(f"Error at ({center_x},{center_y},{center_z}): {str(e)}")
                    continue
    
    # 检查每个条件是否有足够的数据
    valid_conditions = []
    for cond, data_list in condition_data.items():
        if len(data_list) > 0:
            valid_conditions.append(cond)
    
    # 降低条件要求 - 只需要2个条件即可计算相似度
    if len(valid_conditions) < 2:  # 修改为至少2个条件
        if debug_flag:
            print(f"Insufficient conditions at ({center_x},{center_y},{center_z}): {len(valid_conditions)}")
        return None
    
    # 对每个有效条件，计算所有被试的平均响应
    final_condition_data = []
    for cond in sorted(valid_conditions):
        if condition_data[cond]:
            # 对每个被试的数据取平均，得到该条件的整体响应模式
            condition_tensor = torch.stack(condition_data[cond])
            condition_mean = torch.mean(condition_tensor, dim=0)
            final_condition_data.append(condition_mean)
    
    # 合并所有条件数据为一个张量
    window_data = torch.stack(final_condition_data)
    
    # 调试信息
    if debug_flag:
        print(f"Valid window at ({center_x},{center_y},{center_z}) with {len(valid_conditions)} conditions")
    
    return window_data

def statistic_threshold(rdm_volume, alpha=0.05, correction_method="fdr"):
    """
    对搜索光RDM结果进行统计阈值处理
    
    参数:
    rdm_volume: torch.Tensor, 搜索光RDM结果
    alpha: float, 显著性水平
    correction_method: str, 多重比较校正方法: "fdr", "bonferroni", "none"
    
    返回:
    thresholded_volume: torch.Tensor, 阈值化后的RDM结果
    """
    # 提取非零体素的RDM值
    non_zero_mask = torch.any(torch.any(rdm_volume != 0, dim=-1), dim=-1)
    non_zero_rdms = rdm_volume[non_zero_mask]
    
    # 计算每个RDM的统计量（这里简化为非对角线元素的平均值）
    statistic_values = []
    for rdm in non_zero_rdms:
        # 获取非对角线元素
        off_diag = rdm.flatten()[:-1].reshape(rdm.shape[0]-1, rdm.shape[1]+1)[:, 1:].flatten()
        statistic_values.append(torch.mean(off_diag).item())
    
    # 排序统计量
    sorted_values = sorted(statistic_values)
    
    # 应用多重比较校正
    if correction_method == "fdr":
        # FDR校正 (Benjamini-Hochberg)
        m = len(sorted_values)
        thresholds = [(i+1) * alpha / m for i in range(m)]
        
        # 找到最大的满足条件的索引
        threshold_idx = 0
        for i in range(m-1, -1, -1):
            if sorted_values[i] <= thresholds[i]:
                threshold_idx = i
                break
        
        threshold_value = sorted_values[threshold_idx]
        
    elif correction_method == "bonferroni":
        # Bonferroni校正
        m = len(sorted_values)
        threshold_idx = int((alpha / m) * m)
        if threshold_idx < 0:
            threshold_idx = 0
        threshold_value = sorted_values[threshold_idx]
        
    else:  # "none"
        # 不进行校正
        threshold_idx = int(alpha * len(sorted_values))
        threshold_value = sorted_values[threshold_idx]
    
    # 应用阈值
    thresholded_volume = torch.zeros_like(rdm_volume)
    
    # 只保留大于阈值的RDM
    for idx, rdm in enumerate(rdm_volume.reshape(-1, rdm_volume.shape[-2], rdm_volume.shape[-1])):
        flat_idx = idx
        x = flat_idx // (rdm_volume.shape[1] * rdm_volume.shape[2])
        flat_idx %= (rdm_volume.shape[1] * rdm_volume.shape[2])
        y = flat_idx // rdm_volume.shape[2]
        z = flat_idx % rdm_volume.shape[2]
        
        # 计算当前RDM的统计量
        off_diag = rdm.flatten()[:-1].reshape(rdm.shape[0]-1, rdm.shape[1]+1)[:, 1:].flatten()
        rdm_statistic = torch.mean(off_diag).item()
        
        # 如果统计量大于阈值，保留当前RDM
        if rdm_statistic > threshold_value:
            thresholded_volume[x, y, z] = rdm
    
    return thresholded_volume
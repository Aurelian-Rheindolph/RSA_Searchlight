import os
import numpy as np
import torch
import nibabel as nib

# 从各个模块导入函数
from rdm_utils import load_data, load_and_combine_masks, save_rdm_volume, create_brain_map_from_rdms,compute_rdm
from searchlight import searchlight_analysis
from coordinates import mni_to_voxel
from visualization import visualize_pairwise_comparison, visualize_glass_brain, visualize_rdm_with_labels
from visualization import extract_roi_activations, compare_roi_conditions

def main():
    """
    Main function for RDM computation and visualization
    """
    # 数据路径
    data_dir = "../data/firstlevel_forRSA"
    mni_template_path = "../data/MNI152_T1_1mm.nii.gz"

    
    # 创建输出目录
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    
    # 条件标签
    conditions = {
        "positive": "beta_0002.nii",
        "negative": "beta_0004.nii",
        "unrelated": "beta_0006.nii"
    }
    
    # MNI坐标
    mni_coordinates = np.array([
        [15, 12, 66],
        [-66, -21, 15],
        [-48, 36, 24],
        [-24, -12, -24]
    ])
    
    # 搜索光窗口大小
    window_size = [3, 3, 3]
    
    # 加载数据
    print("Loading data...")
    all_data = load_data(data_dir, conditions)

    # 获取真实 affine（第一个被试第一个条件的 affine）
    first_sub = next(iter(all_data))
    first_cond = next(iter(all_data[first_sub]))
    affine = all_data[first_sub][first_cond]["affine"]

    # 加载并集mask
    subject_ids = [f"sub_{i:02d}" for i in range(1, 31)]
    combined_mask = load_and_combine_masks(data_dir, subject_ids)
    
    # 加载数据用于ROI分析
    print("Preparing data for ROI analysis...")
    # 创建[条件数, x, y, z]的平均激活图
    condition_keys = list(conditions.keys())
    n_conditions = len(condition_keys)
    
    # 确定激活图尺寸
    first_data = all_data[first_sub][condition_keys[0]]["data"]
    data_shape = first_data.shape
    
    # 初始化条件平均激活图
    condition_avg_maps = torch.zeros((n_conditions, *data_shape), dtype=torch.float32)
    
    # 计算每个条件的平均图
    print("Computing average activation maps for each condition...")
    for i, cond in enumerate(condition_keys):
        # 收集所有被试该条件的数据
        cond_data = []
        for sub_id in all_data:
            if cond in all_data[sub_id]:
                cond_data.append(all_data[sub_id][cond]["data"])
        # 计算平均
        condition_avg_maps[i] = torch.mean(torch.stack(cond_data), dim=0)
    
    # 执行搜索光分析
    print("Running searchlight RDM analysis...")
    rdm_results = searchlight_analysis(all_data, window_size)
    
    # 将搜索光结果保存为nii文件
    rdm_volume_path = os.path.join(output_dir, "searchlight_rdm_results.nii.gz")
    save_rdm_volume(rdm_results, affine, rdm_volume_path)
    
    # 从RDM体积创建激活图（均值和最大值两种方法）
    print("Creating brain activation maps from RDM results...")
    for method in ['mean', 'max']:
        # 创建脑激活图
        brain_map = create_brain_map_from_rdms(rdm_results, method=method)
        
        # 保存为nii文件
        brain_map_path = os.path.join(output_dir, f"brain_map_{method}.nii.gz")
        
        # 创建NIfTI对象并保存
        nii_img = nib.Nifti1Image(brain_map, affine)
        nib.save(nii_img, brain_map_path)
        print(f"Saved brain activation map ({method}) to {brain_map_path}")
        
        # 可视化激活图
        glass_brain_path = os.path.join(output_dir, f"glass_brain_{method}")
        visualize_glass_brain(
            brain_map, 
            affine=affine, 
            title=f"Brain Activation Map ({method})", 
            save_path=f"{glass_brain_path}.png"
        )
        print(f"Generated glass brain visualizations with {method} method")
    
    # 在指定MNI坐标处计算RDM
    print("Computing RDM at specified MNI coordinates...")
    print("Converting MNI coordinates to voxel coordinates using MNI152 template...")
    # 转换MNI坐标到体素坐标
    try:
        voxel_coordinates = []
        for coord in mni_coordinates:
            voxel_coord = mni_to_voxel(coord, affine=affine, template_path=mni_template_path)
            print(f"MNI coordinate {coord} -> voxel coordinate {voxel_coord}")
            voxel_coordinates.append(voxel_coord)
    except Exception as e:
        print(f"Error in MNI-to-voxel conversion: {str(e)}")
        print("Falling back to default conversion method...")
        voxel_coordinates = [mni_to_voxel(coord, affine=affine) for coord in mni_coordinates]
    
    # 在指定坐标提取RDM（只用mask内体素且去除nan，最大化有效体素数）
    roi_rdms = []
    valid_coords = []
    for i, coord in enumerate(voxel_coordinates):
        x, y, z = coord
        if (x < 0 or y < 0 or z < 0 or 
            x >= rdm_results.shape[0] or 
            y >= rdm_results.shape[1] or 
            z >= rdm_results.shape[2]):
            print(f"Warning: Coordinate {coord} is out of data range")
            continue
        # 只保留mask内体素且无nan的RDM
        rdm = rdm_results[x, y, z]
        if combined_mask[x, y, z] > 0 and not torch.isnan(rdm).any():
            roi_rdms.append(rdm)
            valid_coords.append(i)
        else:
            print(f"Skipping ROI at {coord} (outside mask or contains nan)")
    
    # 可视化每个位置的RDM并保存结果
    print("Visualizing RDMs at ROIs...")
    condition_labels = list(conditions.keys())  # 获取条件标签列表["positive", "negative", "unrelated"]

    for i in range(len(mni_coordinates)):
        if i in valid_coords:
            title = f"MNI coordinate {mni_coordinates[i]} (voxel coordinate {voxel_coordinates[i]})"
            print(f"Visualizing valid RDM at {title}")
            
            # 保存RDM可视化，传递条件标签
            rdm_file_name = f"rdm_roi_{i}_mni_{','.join(map(str, mni_coordinates[i]))}"
            rdm_path = os.path.join(output_dir, f"{rdm_file_name}.png")
            
            # 确保roi_rdms[i]是3×3矩阵
            if roi_rdms[i].shape[0] == 3:
                print(f"ROI {i} has correct RDM shape: {roi_rdms[i].shape}")
            else:
                print(f"WARNING: ROI {i} has unexpected RDM shape: {roi_rdms[i].shape}, expected (3,3)")
            
            # 使用带标签的RDM可视化函数
            visualize_rdm_with_labels(
                roi_rdms[i], 
                title=title, 
                save_path=rdm_path,
                condition_labels=condition_labels
            )
            
            # 为每个有效的ROI创建条件标签
            with open(os.path.join(output_dir, f"{rdm_file_name}_info.txt"), 'w') as f:
                f.write(f"MNI坐标: {mni_coordinates[i]}\n")
                f.write(f"体素坐标: {voxel_coordinates[i]}\n")
                f.write(f"条件标签: {condition_labels}\n")
                
                # 写入RDM的具体值
                f.write("\nRDM martix:\n")
                rdm_np = roi_rdms[i].detach().cpu().numpy()
                f.write("\t" + "\t".join(condition_labels) + "\n")  # 表头
                for row_idx, row in enumerate(rdm_np):
                    row_str = "\t".join([f"{val:.4f}" for val in row])
                    f.write(f"{condition_labels[row_idx]}\t{row_str}\n")
        else:
            print(f"Skipping invalid RDM at MNI coordinate {mni_coordinates[i]}")
    
    # 对每个ROI坐标执行基于ROI的分析
    print("\n=====================")
    print("执行基于ROI的分析...")
    print("=====================")
    
    # 为每个感兴趣区域创建球形ROI
    roi_radius = 5  # mm
    for i, coord in enumerate(voxel_coordinates):
        if i not in valid_coords:
            print(f"跳过无效坐标 {mni_coordinates[i]} (voxel: {coord})")
            continue
            
        print(f"\n分析ROI {i+1}/{len(valid_coords)}: MNI坐标 {mni_coordinates[i]}")
        
        # 创建以该坐标为中心的球形ROI掩码
        roi_mask = np.zeros(data_shape, dtype=bool)
        x, y, z = coord
        
        # 简单球形ROI (可替换为更复杂的解剖学掩码)
        for dx in range(-roi_radius, roi_radius+1):
            for dy in range(-roi_radius, roi_radius+1):
                for dz in range(-roi_radius, roi_radius+1):
                    # 确保是球形 (欧氏距离)
                    if dx**2 + dy**2 + dz**2 <= roi_radius**2:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        # 确保在图像范围内
                        if (0 <= nx < data_shape[0] and 
                            0 <= ny < data_shape[1] and 
                            0 <= nz < data_shape[2]):
                            # 确保在全脑掩码内
                            if combined_mask[nx, ny, nz]:
                                roi_mask[nx, ny, nz] = True
        
        # 检查ROI大小
        roi_size = np.sum(roi_mask)
        if roi_size == 0:
            print(f"警告: ROI在坐标 {coord} 处没有有效体素，请调整半径或坐标")
            continue
        print(f"ROI包含 {roi_size} 个有效体素")
        
        # 保存ROI掩码
        roi_mask_nii = nib.Nifti1Image(roi_mask.astype(np.int8), affine)
        roi_mask_path = os.path.join(output_dir, f"roi_{i}_mni_{'_'.join(map(str, mni_coordinates[i]))}.nii.gz")
        nib.save(roi_mask_nii, roi_mask_path)
        print(f"已保存ROI掩码到 {roi_mask_path}")
        
        # 提取ROI的平均激活向量
        print("提取ROI内的平均激活向量...")
        roi_activations = extract_roi_activations(condition_avg_maps, roi_mask)
        print(f"ROI激活向量形状: {roi_activations.shape}")
        
        # 计算完整的RDM (3x3)
        print("计算完整ROI的RDM...")
        roi_rdm = compute_rdm(roi_activations)
        rdm_path = os.path.join(output_dir, f"roi_{i}_full_rdm.png")
        visualize_rdm_with_labels(
            roi_rdm,
            title=f"ROI {i} 完整RDM: MNI {mni_coordinates[i]}",
            save_path=rdm_path,
            condition_labels=condition_keys
        )
        
        # 计算条件两两比较的RDM
        print("计算ROI内条件两两比较的RDM...")
        condition_pairs = [
            (0, 1),  # positive vs negative
            (0, 2),  # positive vs unrelated
            (1, 2)   # negative vs unrelated
        ]
        
        pairwise_rdms = compare_roi_conditions(
            condition_avg_maps, 
            roi_mask, 
            condition_pairs=condition_pairs,
            condition_labels=condition_keys,
            title=f"ROI {i}: MNI {mni_coordinates[i]}",
            save_path=os.path.join(output_dir, f"roi_{i}_pairwise_rdm.png")
        )
        
        # 输出两两比较结果
        print("\nROI内条件两两比较结果:")
        for pair, rdm in pairwise_rdms.items():
            c1, c2 = pair
            rdm_np = rdm.detach().cpu().numpy()
            dissimilarity = rdm_np[0, 1]  # 获取非对角线值
            print(f"{condition_keys[c1]} vs {condition_keys[c2]}: 相异度 = {dissimilarity:.4f}")
        
        # 导出RDM结果为CSV
        rdm_csv_path = os.path.join(output_dir, f"roi_{i}_rdm_values.csv")
        with open(rdm_csv_path, 'w') as f:
            # 写入标题行
            f.write("ROI,条件1,条件2,相异度\n")
            for pair, rdm in pairwise_rdms.items():
                c1, c2 = pair
                rdm_np = rdm.detach().cpu().numpy()
                dissimilarity = rdm_np[0, 1]
                f.write(f"ROI_{i},{condition_keys[c1]},{condition_keys[c2]},{dissimilarity:.6f}\n")
        print(f"已保存RDM值到 {rdm_csv_path}")
    
    # 执行条件的两两比对分析
    print("Running pairwise condition comparison...")
    pairwise_comparisons = [
        ("positive", "negative"),
        ("positive", "unrelated"),
        ("negative", "unrelated")
    ]
    
    # 计算和存储所有两两比对的RDM矩阵并导出PNG图片
    pairwise_rdms = {}

    # 两两条件比对时也只用mask内体素且去除nan，最大化有效体素数
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
        # 修正：将mask转为torch.Tensor并与cond1_mean同设备
        mask_idx = torch.tensor(mask > 0, dtype=torch.bool, device=cond1_mean.device)
        valid_idx = mask_idx & (~torch.isnan(cond1_mean)) & (~torch.isnan(cond2_mean))
        cond1_masked = cond1_mean[valid_idx]
        cond2_masked = cond2_mean[valid_idx]
        if cond1_masked.numel() == 0 or cond2_masked.numel() == 0:
            raise ValueError("No valid voxels for pairwise RDM!")
        combined = torch.stack([cond1_masked, cond2_masked])
        return compute_rdm(combined)

    for cond1, cond2 in pairwise_comparisons:
        print(f"Computing RDM matrix for {cond1} vs {cond2}...")
        try:
            # 计算条件对比RDM
            pairwise_rdms[(cond1, cond2)] = masked_pairwise_rdm(all_data, cond1, cond2, combined_mask)
            # 输出RDM矩阵
            print(f"RDM for {cond1} vs {cond2}:\n{pairwise_rdms[(cond1, cond2)].detach().cpu().numpy()}")
            # 导出PNG图片
            save_path = os.path.join(output_dir, f"pairwise_{cond1}_vs_{cond2}.png")
            visualize_pairwise_comparison(
                pairwise_rdms[(cond1, cond2)],
                f"{cond1} vs {cond2}",
                save_path=save_path,
                condition_labels=[cond1, cond2]
            )
            print(f"Saved pairwise RDM visualization for {cond1} vs {cond2} to {save_path}")
        except Exception as e:
            print(f"Error computing RDM for {cond1} vs {cond2}: {str(e)}")
            pairwise_rdms[(cond1, cond2)] = None
    
    print(f"Generated {len(pairwise_rdms)} pairwise RDM matrices")
    
    # 创建一个表格汇总所有条件对比的RDM结果
    with open(os.path.join(output_dir, "pairwise_comparison_summary.tsv"), 'w') as f:
        f.write("Condition1\tCondition2\tDissimilarity\n")
        
        for cond1, cond2 in pairwise_comparisons:
            print(f"Comparing {cond1} vs {cond2}...")
            try:
                # 计算条件对比RDM
                pairwise_rdm = masked_pairwise_rdm(all_data, cond1, cond2, combined_mask)
                
                # 保存RDM可视化，传递条件标签
                save_path = os.path.join(output_dir, f"pairwise_{cond1}_vs_{cond2}.png")
                visualize_pairwise_comparison(
                    pairwise_rdm, 
                    f"{cond1} vs {cond2}", 
                    save_path=save_path,
                    condition_labels=[cond1, cond2]  # 传递条件标签
                )
                
                # 写入表格
                rdm_np = pairwise_rdm.detach().cpu().numpy()
                dissimilarity = rdm_np[0, 1]  # 获取非对角线值
                f.write(f"{cond1}\t{cond2}\t{dissimilarity:.4f}\n")
            except Exception as e:
                print(f"Error comparing {cond1} vs {cond2}: {str(e)}")
                f.write(f"{cond1}\t{cond2}\tERROR\n")
    
    # 创建一个README文件，说明分析结果
    with open(os.path.join(output_dir, "README.txt"), 'w') as f:
        f.write("RSA分析结果说明\n")
        f.write("==============\n\n")
        f.write("本目录包含以下内容：\n\n")
        f.write("1. 搜索光RDM结果\n")
        f.write("   - searchlight_rdm_results.nii.gz: 包含整个脑体积中每个体素的RDM信息\n")
        f.write("   - brain_map_mean.nii.gz: 基于RDM平均值创建的脑激活图\n")
        f.write("   - brain_map_max.nii.gz: 基于RDM最大值创建的脑激活图\n")
        f.write("   - glass_brain_*.png: 玻璃脑可视化，显示了哪些区域有显著的RDM模式\n\n")
        
        f.write("2. ROI分析结果\n")
        for i, mni_coord in enumerate(mni_coordinates):
            coord_str = ','.join(map(str, mni_coord))
            f.write(f"   - rdm_roi_{i}_mni_{coord_str}.png: MNI坐标{mni_coord}处的RDM可视化\n")
        f.write("\n")
        
        f.write("3. 条件对比结果\n")
        for cond1, cond2 in pairwise_comparisons:
            f.write(f"   - pairwise_{cond1}_vs_{cond2}.png: {cond1}和{cond2}条件之间的RDM比较\n")
        f.write("   - pairwise_comparison_summary.tsv: 所有条件对比的相异度汇总表\n\n")
        
        f.write("分析参数：\n")
        f.write(f"- 搜索光窗口大小: {window_size}\n")
        f.write(f"- 条件数量: {len(conditions)}\n")
        f.write(f"- 条件标签: {list(conditions.keys())}\n")
        f.write(f"- 数据路径: {data_dir}\n")
    
    print("Analysis completed! All results saved to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    main()
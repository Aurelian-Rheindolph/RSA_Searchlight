import os
import numpy as np
from rdm_torch import compute_rdm, searchlight_rdm, mni_to_voxel, extract_roi_window, normalize_rdm
from visualization import plot_glass_brain, plot_rdm, save_searchlight_roi_results, visualize_searchlight_heatmap, export_roi_searchlight_results, visualize_roi_rdms
from utils import load_nifti_file, load_and_combine_subjects, create_output_directory, get_roi_coordinates


def main():
    print("=== RSA Pipeline Start ===")
    data_path = os.path.join('data', 'firstlevel_forRSA')
    output_path = 'outputs'
    create_output_directory(output_path)
    subjects = [f'sub_{i:02d}' for i in range(1, 31)]
    beta_names = ['beta_0002.nii', 'beta_0004.nii', 'beta_0006.nii']
    condition_names = ['positive', 'negative', 'unrelated']

    print("[1/7] Loading all subjects' beta data...")
    all_data, _, affine = load_and_combine_subjects(data_path, subject_prefix='sub_', beta_files=beta_names, mask_file=None)
    print("[2/7] Calculating mean beta across subjects...")
    fmri_data = np.nanmean(all_data, axis=0)  # (3, x, y, z)
    print("fmri_data stats:", "min", fmri_data.min(), "max", fmri_data.max(), "mean", fmri_data.mean())
    print("all_data shape:", all_data.shape)

    print("[3/7] Computing whole-brain RDM and visualizing...")
    rdm = compute_rdm(fmri_data)

    # 保存原始RDM（未归一化）
    np.save(os.path.join(output_path, 'raw_roi_rdm.npy'), rdm)
    # 可视化时，同时保存归一化和非归一化的版本
    plot_rdm(rdm, condition_names, title="Whole-brain RDM (Raw)", 
             output_path=os.path.join(output_path, 'raw_roi_rdm.png'))
    # 归一化版本用于比较
    normalized_rdm = normalize_rdm(rdm)
    plot_rdm(normalized_rdm, condition_names, title="Whole-brain RDM (Normalized)", 
             output_path=os.path.join(output_path, 'normalized_roi_rdm.png'))
    print("    Whole-brain RDM saved to outputs/raw_roi_rdm.png and outputs/normalized_roi_rdm.png")

    print("[4/7] Running whole-brain searchlight (3 conditions)...")
    # 对每个受试者进行searchlight分析，然后求平均
    all_subj_rdm_maps = []
    for subj in subjects:
        print(f"    Running searchlight for subject {subj}...")
        subj_path = os.path.join(data_path, subj)
        beta_files = [os.path.join(subj_path, f) for f in beta_names]
        rdm_map = searchlight_rdm(beta_files, None, window_size=(5,5,5))
        all_subj_rdm_maps.append(rdm_map)

    # 计算所有受试者的平均searchlight RDM
    print("    Averaging searchlight RDM maps across all subjects...")
    mean_rdm_map = np.nanmean(all_subj_rdm_maps, axis=0)
    
    # 保存平均后的结果
    save_searchlight_roi_results(mean_rdm_map, None, threshold=0.5, output_dir='outputs')
    print("    Searchlight RDM (3 conditions) for all subjects completed.")

    # Generate heatmap glass brain (with and without threshold versions)
    visualize_searchlight_heatmap(mean_rdm_map, affine, threshold=0, 
                                 output_prefix='searchlight_heatmap', 
                                 output_dir='outputs')
    
    # 直接使用plot_glass_brain可视化NIfTI文件
    nii_path = os.path.join('outputs', 'searchlight_heatmap.nii.gz')
    plot_glass_brain(
        nii_path,
        title='Searchlight RDM Results (30 Subjects Average)',
        output_path=os.path.join('outputs', 'direct_glass_brain_visualization.png'),
        threshold=0.3,
        colorbar=True,
        cmap='coolwarm',
        display_mode='lyrz',  # 四视图模式：左侧、右侧、底视图和顶视图
        vmin=None,  # 使用原始数据范围，不进行归一化
        vmax=None   # 使用原始数据范围，不进行归一化
    )
    print("    Direct glass brain visualization saved to outputs/direct_glass_brain_visualization.png")

    print("[5/7] Running pairwise searchlight comparisons...")
    pair_names = [
        ('positive', 'unrelated', 'beta_0002.nii', 'beta_0006.nii'),
        ('negative', 'unrelated', 'beta_0004.nii', 'beta_0006.nii'),
        ('positive', 'negative', 'beta_0002.nii', 'beta_0004.nii')
    ]
    for cond1, cond2, beta1, beta2 in pair_names:
        print(f"    Pairwise: {cond1} vs {cond2} ...")
        all_pair_rdm = []
        for subj in subjects:
            subj_path = os.path.join(data_path, subj)
            beta_files = [os.path.join(subj_path, beta1), os.path.join(subj_path, beta2)]
            rdm_map = searchlight_rdm(beta_files, None, window_size=(5,5,5))
            all_pair_rdm.append(rdm_map)
        mean_rdm_map = np.mean(all_pair_rdm, axis=0)
        max_idx = np.unravel_index(np.nanargmax(mean_rdm_map[0,1]), mean_rdm_map[0,1].shape)
        rdm_2x2 = np.array([[0, mean_rdm_map[0,1][max_idx]], [mean_rdm_map[1,0][max_idx], 0]])
        
        # 对绘图函数传递参数，让其使用自定义的vmin和vmax范围，确保可视化正常
        plot_rdm(rdm_2x2, [cond1, cond2], title=f'RDM: {cond1} vs {cond2} (Unnormalized)', 
                output_path=os.path.join(output_path, f'rdm_{cond1}_vs_{cond2}.png'))
        print(f"        Max activation at voxel {max_idx}, RDM value: {rdm_2x2[0,1]:.4f}, saved to outputs/rdm_{cond1}_vs_{cond2}.png")

    # 获取ROI坐标和名称信息，并转换为体素坐标
    roi_info = get_roi_coordinates()
    roi_coordinates = []
    
    print("[6/7] Exporting searchlight results for specified ROI coordinates...")
    for idx, (mni, roi_name) in enumerate(roi_info):
        voxel = mni_to_voxel(mni, affine)
        roi_coordinates.append((mni, voxel, roi_name))
        print(f"    ROI {idx+1}: {roi_name} - MNI {mni} -> Voxel Coordinates {voxel}")
    
    # 导出所有ROI的玻璃脑图像和坐标信息
    export_roi_searchlight_results(roi_coordinates, affine, output_dir=output_path)
    print("    Glass brain images and coordinate information for all ROIs have been exported to respective directories")
    
    # 为每个ROI计算并可视化RDM
    roi_rdms = {}
    roi_raw_rdms = {}
    for mni, voxel, roi_name in roi_coordinates:
        # 为每个ROI计算RDM
        all_subj_rdms = []
        for subj in subjects:
            subj_path = os.path.join(data_path, subj)
            beta_files = [os.path.join(subj_path, f) for f in beta_names]
            
            # 读取beta数据
            subj_betas = []
            for beta_file in beta_files:
                beta_data, _ = load_nifti_file(beta_file)
                subj_betas.append(beta_data)
            subj_betas = np.stack(subj_betas, axis=0)  # (conditions, x, y, z)
            
            # 从体素坐标提取ROI窗口数据
            roi_data = extract_roi_window(subj_betas, voxel, size=3)  # shape: (conditions, N)
            
            # 计算该ROI的RDM（不归一化）
            subj_rdm = compute_rdm(roi_data)
            all_subj_rdms.append(subj_rdm)
        
        # 计算平均RDM（保留原始值，不进行归一化）
        mean_rdm = np.nanmean(all_subj_rdms, axis=0)
        roi_raw_rdms[roi_name] = mean_rdm
        
        # 同时保存归一化版本用于可视化比较
        roi_rdms[roi_name] = normalize_rdm(mean_rdm)
    
    # 可视化所有ROI的RDM（原始值）
    for idx, (mni, voxel, roi_name) in enumerate(roi_coordinates):
        roi_dir = os.path.join(output_path, f"ROI_{idx+1}_{roi_name}")
        os.makedirs(roi_dir, exist_ok=True)
        
        if roi_name in roi_raw_rdms:
            raw_rdm = roi_raw_rdms[roi_name]
            
            # 保存原始RDM矩阵为npy文件
            raw_rdm_path = os.path.join(roi_dir, f'raw_rdm_{roi_name}.npy')
            np.save(raw_rdm_path, raw_rdm)
            
            # 可视化原始RDM
            plot_rdm(
                raw_rdm, 
                condition_names, 
                title=f'ROI {idx+1}: {roi_name} Raw RDM', 
                output_path=os.path.join(roi_dir, f'raw_rdm_{roi_name}.png')
            )
            
            # 同时保存归一化版本用于比较
            norm_rdm = roi_rdms[roi_name]
            norm_rdm_path = os.path.join(roi_dir, f'normalized_rdm_{roi_name}.npy')
            np.save(norm_rdm_path, norm_rdm)
            
            plot_rdm(
                norm_rdm,
                condition_names,
                title=f'ROI {idx+1}: {roi_name} Normalized RDM',
                output_path=os.path.join(roi_dir, f'normalized_rdm_{roi_name}.png')
            )
    
    print("    RDMs for all ROIs have been computed and visualized (both raw and normalized versions)")
    
    # 在searchlight_rdm结果中提取指定ROI的结果
    print("[7/7] Extracting searchlight RDM results for specified ROIs...")
    
    # 对每个ROI执行searchlight分析
    for idx, (mni, voxel, roi_name) in enumerate(roi_coordinates):
        roi_dir = os.path.join(output_path, f"ROI_{idx+1}_{roi_name}")
        os.makedirs(roi_dir, exist_ok=True)
        
        # 创建该ROI的mask
        mask = np.zeros(fmri_data.shape[1:])
        if 0 <= voxel[0] < mask.shape[0] and 0 <= voxel[1] < mask.shape[1] and 0 <= voxel[2] < mask.shape[2]:
            mask[voxel] = 1
        
        # 对该ROI执行searchlight，使用所有受试者数据
        print(f"    Performing searchlight analysis for ROI {idx+1}: {roi_name}...")
        
        # 对每个受试者进行ROI searchlight分析，然后求平均
        roi_rdm_maps = []
        for subj in subjects:
            subj_path = os.path.join(data_path, subj)
            beta_files = [os.path.join(subj_path, f) for f in beta_names]
            subj_rdm_roi = searchlight_rdm(beta_files, mask_data=mask, window_size=(5,5,5))
            roi_rdm_maps.append(subj_rdm_roi)
        
        # 计算所有受试者的平均ROI searchlight RDM
        rdm_roi = np.nanmean(roi_rdm_maps, axis=0)
        
        # 保存结果
        rdm_npy_path = os.path.join(roi_dir, f'searchlight_rdm_{roi_name}.npy')
        np.save(rdm_npy_path, rdm_roi)
        
        # 提取该体素位置的RDM并可视化
        if 0 <= voxel[0] < rdm_roi.shape[2] and 0 <= voxel[1] < rdm_roi.shape[3] and 0 <= voxel[2] < rdm_roi.shape[4]:
            roi_rdm = rdm_roi[:, :, voxel[0], voxel[1], voxel[2]]
            roi_rdm = normalize_rdm(roi_rdm)
            
            plot_rdm(
                roi_rdm, 
                condition_names, 
                title=f'ROI {idx+1}: {roi_name} Searchlight RDM (30 Subjects Average)', 
                output_path=os.path.join(roi_dir, f'searchlight_rdm_{roi_name}.png')
            )
        
        # 可视化该ROI的searchlight结果热力图
        nii_path, _ = visualize_searchlight_heatmap(
            rdm_roi, 
            affine=affine,
            threshold=0.3,
            output_prefix=f'roi_{idx+1}_{roi_name}_heatmap',
            output_dir=roi_dir
        )
        print(f"    Searchlight results for ROI {idx+1}: {roi_name} have been saved to {roi_dir}")
    
    print("=== RSA Pipeline Finished ===")

if __name__ == "__main__":
    main()
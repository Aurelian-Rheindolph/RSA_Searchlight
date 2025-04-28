import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def visualize_rdm(rdm, title=None, save_path=None):
    # 转换torch张量为numpy数组
    if isinstance(rdm, torch.Tensor):
        rdm_np = rdm.detach().cpu().numpy()
    else:
        rdm_np = rdm
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 确保对角线元素为0
    np.fill_diagonal(rdm_np, 0)
    
    # 不再使用mask屏蔽非对角线元素，显示完整矩阵
    sns.heatmap(
        rdm_np,
        cmap='coolwarm',
        square=True,
        vmin=0,  # 相异度从0开始
        vmax=max(0.1, rdm_np.max()),  # 上限设为最大值或至少0.1
        annot=True,  # 在单元格中显示数值
        fmt='.2f',   # 格式化为2位小数
        cbar_kws={'label': 'dissimilarity (1-r)'}
    )
    
    # 添加标题
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_rdm_brain(rdm, title=None, save_path=None):
    # 转换torch张量为numpy数组
    if isinstance(rdm, torch.Tensor):
        rdm_np = rdm.detach().cpu().numpy()
    else:
        rdm_np = rdm
    
    # 创建图形
    plt.figure(figsize=(12, 5))
    
    # 左侧：RDM矩阵热图
    plt.subplot(1, 2, 1)
    
    # 确保对角线元素为0
    np.fill_diagonal(rdm_np, 0)
    
    # 绘制完整热图，不再使用mask屏蔽对角线
    sns.heatmap(
        rdm_np,
        cmap='coolwarm',
        square=True,
        vmin=0,
        vmax=max(0.1, rdm_np.max()),  # 确保颜色范围合适
        annot=True,  # 在单元格中显示数值
        fmt='.2f',   # 格式化为2位小数
        cbar_kws={'label': '相异度 (1-r)'}
    )
    plt.title("Representational Dissimilarity Matrix (RDM)")
    
    # 右侧：MDS条件相异度可视化
    plt.subplot(1, 2, 2)
    
    # 使用MDS进行降维
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    
    # 确保RDM是对称的
    rdm_symmetric = (rdm_np + rdm_np.T) / 2
    
    # 计算MDS坐标
    points = mds.fit_transform(rdm_symmetric)
    
    # 绘制MDS散点图
    condition_labels = [f"条件{i+1}" for i in range(rdm_np.shape[0])]
    plt.scatter(points[:, 0], points[:, 1], s=100)
    
    # 添加条件标签
    for i, label in enumerate(condition_labels):
        plt.annotate(label, (points[i, 0], points[i, 1]), fontsize=10,
                     xytext=(5, 5), textcoords='offset points')
    
    plt.title("MDS Visualization")
    plt.axis('equal')
    
    # 添加整体标题
    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def create_brain_map_from_rdms(rdm_volume, method='mean'):
    # Convert torch tensor to numpy array
    if isinstance(rdm_volume, torch.Tensor):
        rdm_volume_np = rdm_volume.detach().cpu().numpy()
    else:
        rdm_volume_np = rdm_volume
    
    # Extract shape of the RDM volume excluding the last two dimensions
    spatial_shape = rdm_volume_np.shape[:-2]
    
    # Create brain activation map
    brain_map = np.zeros(spatial_shape)
    
    # Calculate statistics for each voxel's RDM
    if method == 'mean':
        for i in range(spatial_shape[0]):
            for j in range(spatial_shape[1]):
                for k in range(spatial_shape[2]):
                    rdm = rdm_volume_np[i, j, k]
                    # Calculate mean of non-diagonal elements
                    mask = ~np.eye(rdm.shape[0], dtype=bool)
                    brain_map[i, j, k] = np.mean(rdm[mask])
    
    elif method == 'max':
        for i in range(spatial_shape[0]):
            for j in range(spatial_shape[1]):
                for k in range(spatial_shape[2]):
                    rdm = rdm_volume_np[i, j, k]
                    # Calculate maximum of non-diagonal elements
                    mask = ~np.eye(rdm.shape[0], dtype=bool)
                    brain_map[i, j, k] = np.max(rdm[mask])
    
    return brain_map

def visualize_glass_brain(brain_map, affine=None, title=None, save_path=None, display_mode='ortho', bg_img=None):
    # 如果没有提供仿射矩阵，使用默认矩阵
    if affine is None:
        affine = np.array([
            [-1, 0, 0, 90],
            [0, 1, 0, -126],
            [0, 0, 1, -72],
            [0, 0, 0, 1]
        ])
    
    nii_img = nib.Nifti1Image(brain_map, affine)
    cmap = plotting.cm.cold_hot
    if bg_img is None or bg_img == 'MNI152':
        mni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/MNI152_T1_1mm.nii.gz'))
        bg_img = mni_path
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        base_path = os.path.splitext(save_path)[0]
        modes = ['x', 'y', 'z', 'ortho']
        for mode in modes:
            # 使用cut_coords=None让nilearn自动选择切片坐标
            plot_kwargs = {
                'display_mode': mode,
                'colorbar': True,
                'title': f"{title} - {mode.upper()} view" if title else f"{mode.upper()} view",
                'threshold': 0.0,
                'cmap': cmap,
                'annotate': True,
                'vmax': None,
                'resampling_interpolation': 'cubic',
                'bg_img': bg_img,
                'cut_coords': None  # 关键：自适应坐标轴
            }
            fig = plt.figure(figsize=(24, 18), dpi=600)
            if mode == 'ortho':
                display = plotting.plot_glass_brain(
                    nii_img,
                    figure=fig,
                    alpha=0.6,
                    **plot_kwargs
                )
                display.add_edges(nii_img)
            else:
                plotting.plot_stat_map(
                    nii_img,
                    figure=fig,
                    **plot_kwargs
                )
            mode_path = f"{base_path}_{mode}.png"
            plt.savefig(mode_path, bbox_inches='tight', dpi=600)
            plt.close()
        # 创建带解剖背景的MNI视图（标准视图）
        fig = plt.figure(figsize=(24, 18), dpi=600)
        plot_kwargs_mni = plot_kwargs.copy()
        plot_kwargs_mni['display_mode'] = 'ortho'
        plot_kwargs_mni['cut_coords'] = None  # 关键：自适应坐标轴
        plot_kwargs_mni['title'] = title
        plotting.plot_stat_map(
            nii_img,
            figure=fig,
            **plot_kwargs_mni
        )
        mni_path = f"{base_path}_mni.png"
        plt.savefig(mni_path, bbox_inches='tight', dpi=600)
        plt.close()
        # 添加水平切片视图（此处仍可自定义cut_coords）
        horizontal_cuts = list(np.linspace(-50, 80, 16))
        fig = plt.figure(figsize=(24, 18), dpi=600)
        plot_kwargs_horizontal = plot_kwargs.copy()
        plot_kwargs_horizontal['display_mode'] = 'z'
        plot_kwargs_horizontal['cut_coords'] = horizontal_cuts
        plot_kwargs_horizontal['title'] = f"{title} - 水平切片"
        plotting.plot_stat_map(
            nii_img,
            figure=fig,
            **plot_kwargs_horizontal
        )
        horizontal_path = f"{base_path}_horizontal_slices.png"
        plt.savefig(horizontal_path, bbox_inches='tight', dpi=600)
        plt.close()
        print(f"已保存高分辨率玻璃脑可视化结果到: {base_path}_*.png")
    else:
        # 直接显示时，cut_coords=None自适应
        plot_kwargs = {
            'display_mode': display_mode,
            'colorbar': True,
            'title': title,
            'threshold': 0.0,
            'cmap': cmap,
            'symmetric_cbar': True,
            'resampling_interpolation': 'cubic',
            'bg_img': bg_img,
            'cut_coords': None  # 关键：自适应坐标轴
        }
        fig = plt.figure(figsize=(24, 18), dpi=300)
        plotting.plot_glass_brain(nii_img, figure=fig, alpha=0.8, **plot_kwargs)
        plt.show()

def visualize_pairwise_comparison(pairwise_rdm, title=None, save_path=None, condition_labels=None):

    # 转换为numpy数组
    if isinstance(pairwise_rdm, torch.Tensor):
        rdm_np = pairwise_rdm.detach().cpu().numpy()
    else:
        rdm_np = pairwise_rdm
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 设置默认条件标签
    if condition_labels is None:
        if (rdm_np.shape[0] == 2):  # 如果是2x2矩阵
            condition_labels = ["Condition 1", "Condition 2"]
        else:
            condition_labels = [f"Condition {i+1}" for i in range(rdm_np.shape[0])]
    
    # 对于2x2矩阵的特殊处理，不屏蔽对角线
    if rdm_np.shape[0] == 2:
        # 绘制完整的热图
        sns.heatmap(
            rdm_np,
            cmap='coolwarm',
            square=True,
            vmin=0,
            vmax=1,
            annot=True,   # 在单元格中显示数值
            fmt='.2f',    # 格式化为2位小数
            xticklabels=condition_labels,
            yticklabels=condition_labels,
            cbar_kws={'label': 'Dissimilarity (1-r)'}
        )
        
        # 添加对角线元素的解释
        plt.text(0.5, -0.15, "对角线元素表示条件内相似度 (应为0)", 
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=10, style='italic')
        plt.text(0.5, -0.2, "非对角线元素表示条件间差异度", 
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=10, style='italic')
    else:
        # 对于较大的矩阵，屏蔽对角线
        mask = np.zeros_like(rdm_np, dtype=bool)
        np.fill_diagonal(mask, True)  # 屏蔽对角线
        
        # 绘制热图
        sns.heatmap(
            rdm_np,
            cmap='coolwarm',
            mask=mask,
            square=True,
            vmin=0,
            vmax=1,
            annot=True,   # 在单元格中显示数值
            fmt='.2f',    # 格式化为2位小数
            xticklabels=condition_labels,
            yticklabels=condition_labels,
            cbar_kws={'label': 'Between-condition Dissimilarity'}
        )
    
    # 添加标题
    if title:
        plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_rdm_comparison(rdm1, rdm2, labels=None, title=None, save_path=None):
    # Convert to numpy arrays
    if isinstance(rdm1, torch.Tensor):
        rdm1_np = rdm1.detach().cpu().numpy()
    else:
        rdm1_np = rdm1
    
    if isinstance(rdm2, torch.Tensor):
        rdm2_np = rdm2.detach().cpu().numpy()
    else:
        rdm2_np = rdm2
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # RDM1 heatmap
    plt.subplot(1, 3, 1)
    mask1 = np.zeros_like(rdm1_np, dtype=bool)
    np.fill_diagonal(mask1, True)
    sns.heatmap(
        rdm1_np, 
        cmap='coolwarm', 
        mask=mask1, 
        square=True,
        vmin=0, 
        vmax=1,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Dissimilarity (1-r)'}
    )
    plt.title("RDM 1")
    
    # RDM2 heatmap
    plt.subplot(1, 3, 2)
    mask2 = np.zeros_like(rdm2_np, dtype=bool)
    np.fill_diagonal(mask2, True)
    sns.heatmap(
        rdm2_np, 
        cmap='coolwarm', 
        mask=mask2, 
        square=True,
        vmin=0, 
        vmax=1,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Dissimilarity (1-r)'}
    )
    plt.title("RDM 2")
    
    # RDM difference heatmap
    plt.subplot(1, 3, 3)
    diff = rdm1_np - rdm2_np
    mask_diff = np.zeros_like(diff, dtype=bool)
    np.fill_diagonal(mask_diff, True)
    
    # Calculate max absolute difference for symmetric color range
    max_abs_diff = min(max(abs(diff[~mask_diff])), 1.0)
    
    sns.heatmap(
        diff, 
        cmap='coolwarm', 
        mask=mask_diff, 
        square=True,
        center=0, 
        vmin=-max_abs_diff, 
        vmax=max_abs_diff,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Difference (RDM1 - RDM2)'}
    )
    plt.title("RDM Difference")
    
    # Add overall title
    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # Save or display image
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_rdm_with_labels(rdm, title=None, save_path=None, condition_labels=None):
    # 转换为numpy数组
    if isinstance(rdm, torch.Tensor):
        rdm_np = rdm.detach().cpu().numpy()
    else:
        rdm_np = rdm
    
    # 打印RDM值用于调试
    print(f"Visualizing RDM with shape: {rdm_np.shape}")
    print(f"RDM values:\n{rdm_np}")
    
    # 创建图形
    plt.figure(figsize=(16, 7))
    
    # 左侧：使用条件标签的热图
    plt.subplot(1, 2, 1)
    
    # 使用默认标签
    if condition_labels is None or len(condition_labels) != rdm_np.shape[0]:
        condition_labels = [f"Condition {i+1}" for i in range(rdm_np.shape[0])]
    
    # 确保RDM中对角线元素为0
    np.fill_diagonal(rdm_np, 0)
    
    # 绘制热图，不再使用mask屏蔽对角线
    sns.heatmap(
        rdm_np,
        cmap='coolwarm',
        square=True,
        vmin=0,
        vmax=max(0.1, rdm_np.max()),  # 确保颜色范围合适
        annot=True,  # 在单元格中显示数值
        fmt='.2f',   # 格式化为2位小数
        xticklabels=condition_labels,
        yticklabels=condition_labels,
        cbar_kws={'label': 'Dissimilarity (1-r)'}
    )
    plt.title("Representational Dissimilarity Matrix (RDM)")
    
    # 右侧：MDS条件相异度可视化
    plt.subplot(1, 2, 2)
    
    # 使用MDS进行降维
    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        
        # 确保RDM是对称的
        rdm_symmetric = (rdm_np + rdm_np.T) / 2
        
        # 计算MDS坐标
        points = mds.fit_transform(rdm_symmetric)
        
        # 绘制MDS散点图
        plt.scatter(points[:, 0], points[:, 1], s=100)
        
        # 添加条件标签
        for i, label in enumerate(condition_labels):
            plt.annotate(label, (points[i, 0], points[i, 1]), fontsize=12,
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title("多维缩放 (MDS)")
        plt.axis('equal')
    except Exception as e:
        plt.text(0.5, 0.5, f"MDS可视化失败: {str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        print(f"MDS visualization error: {str(e)}")
    
    # 添加整体标题
    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved RDM visualization to: {save_path}")
    else:
        plt.show()

def extract_roi_activations(data, mask, conditions=None, variance_threshold=0.0):
    """
    提取ROI激活，并筛选高方差体素以增强条件间差异
    
    参数:
    variance_threshold: 体素方差阈值，仅保留方差大于此值的体素
    """
    # 转换torch张量为numpy数组
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
        
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = mask.astype(bool)
    
    # 确定数据维度
    if len(data_np.shape) == 4:  # [条件数, x, y, z]
        n_conditions = data_np.shape[0]
        has_subjects = False
    elif len(data_np.shape) == 5:  # [被试数, 条件数, x, y, z]
        n_subjects = data_np.shape[0]
        n_conditions = data_np.shape[1]
        has_subjects = True
    else:
        raise ValueError(f"数据维度不正确: {data_np.shape}，期望 [条件数, x, y, z] 或 [被试数, 条件数, x, y, z]")
    
    # 选择条件
    if conditions is None:
        conditions = list(range(n_conditions))
        
    # 提取ROI内的激活向量
    if has_subjects:
        # [被试数, 条件数, ROI体素数] -> [被试数, 条件数]（对每个体素做平均）
        roi_activations = np.zeros((n_subjects, len(conditions)))
        for s in range(n_subjects):
            for i, c in enumerate(conditions):
                # 提取该被试该条件下的ROI体素值，并对体素做平均
                roi_activations[s, i] = np.mean(data_np[s, c][mask_np])
    else:
        # [条件数, ROI体素数]
        roi_activations = np.zeros((len(conditions), np.sum(mask_np)))
        for i, c in enumerate(conditions):
            # 提取该条件下的ROI体素值
            values = data_np[c][mask_np]
            
            # 标准化体素值
            if standardize:
                values = (values - np.mean(values)) / (np.std(values) + 1e-10)
                
            roi_activations[i] = values
    
    # 计算每个体素在条件间的方差
    if has_subjects:
        # 暂时保留所有体素
        temp_activations = np.zeros((len(conditions), np.sum(mask_np)))
        for i, c in enumerate(conditions):
            # 计算所有被试该条件的平均
            cond_avg = np.mean([data_np[s, c][mask_np] for s in range(n_subjects)], axis=0)
            temp_activations[i] = cond_avg
        
        # 计算每个体素在条件间的方差
        voxel_variance = np.var(temp_activations, axis=0)
        
        # 选择高方差体素（信号对比度高的体素）
        if variance_threshold > 0:
            high_var_voxels = voxel_variance > variance_threshold
            if np.sum(high_var_voxels) > 10:  # 确保至少有10个体素
                # 重新获取仅包含高方差体素的激活数据
                roi_activations = np.zeros((n_subjects, len(conditions)))
                for s in range(n_subjects):
                    for i, c in enumerate(conditions):
                        voxel_values = data_np[s, c][mask_np][high_var_voxels]
                        roi_activations[s, i] = np.mean(voxel_values)
            else:
                print(f"警告：使用方差阈值 {variance_threshold} 筛选后的体素数量不足")
    else:
        # 对于无被试数据，类似处理
        # 计算每个体素在条件间的方差
        voxel_variance = np.var(roi_activations, axis=0)
        
        # 选择高方差体素
        if variance_threshold > 0:
            high_var_voxels = voxel_variance > variance_threshold
            if np.sum(high_var_voxels) > 10:  # 确保至少有10个体素
                # 重新获取仅包含高方差体素的激活数据
                filtered_activations = np.zeros((len(conditions), np.sum(high_var_voxels)))
                for i, c in enumerate(conditions):
                    filtered_activations[i] = roi_activations[i][high_var_voxels]
                roi_activations = filtered_activations
            else:
                print(f"警告：使用方差阈值 {variance_threshold} 筛选后的体素数量不足")
    
    return roi_activations

def compare_roi_conditions(data, mask, condition_pairs=None, condition_labels=None, method='correlation', 
                          title=None, save_path=None, visualize=True):
    import rdm_utils  # 导入RDM计算函数
    
    # 转换torch张量为numpy数组
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    
    # 确定数据维度和条件数
    if len(data_np.shape) == 4:  # [条件数, x, y, z]
        n_conditions = data_np.shape[0]
    elif len(data_np.shape) == 5:  # [被试数, 条件数, x, y, z]
        n_conditions = data_np.shape[1]
    else:
        raise ValueError(f"数据维度不正确: {data_np.shape}，期望 [条件数, x, y, z] 或 [被试数, 条件数, x, y, z]")
    
    # 提取ROI内的激活向量
    roi_activations = extract_roi_activations(data_np, mask)
    
    # 确定要比较的条件对
    if condition_pairs is None:
        condition_pairs = []
        for i in range(n_conditions):
            for j in range(i+1, n_conditions):
                condition_pairs.append((i, j))
    
    # 计算条件对的RDM
    pairwise_rdms = {}
    for pair in condition_pairs:
        c1, c2 = pair
        # 提取两个条件的激活向量
        if len(roi_activations.shape) == 2:  # [条件数, ROI体素数]
            act1 = roi_activations[c1:c1+1]  # 保持维度
            act2 = roi_activations[c2:c2+1]  # 保持维度
            acts = np.vstack([act1, act2])
        else:  # [被试数, 条件数]
            acts = roi_activations[:, [c1, c2]]  # 选择两个条件
            
        # 计算RDM
        rdm = rdm_utils.compute_rdm(acts, method=method)
        pairwise_rdms[pair] = rdm
        
        # 可视化结果
        if visualize:
            pair_labels = None
            if condition_labels and len(condition_labels) >= max(c1, c2) + 1:
                pair_labels = [condition_labels[c1], condition_labels[c2]]
                
            pair_title = f"{pair_labels[0]} vs {pair_labels[1]}" if pair_labels else f"条件 {c1+1} vs 条件 {c2+1}"
            if title:
                pair_title = f"{title}: {pair_title}"
                
            pair_save_path = None
            if save_path:
                base_path = os.path.splitext(save_path)[0]
                pair_save_path = f"{base_path}_pair_{c1+1}_{c2+1}.png"
                
            visualize_pairwise_comparison(rdm, title=pair_title, 
                                          save_path=pair_save_path, 
                                          condition_labels=pair_labels)
    
    # 返回所有条件对的RDM
    return pairwise_rdms
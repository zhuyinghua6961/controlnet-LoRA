import numpy as np
from scipy.ndimage import gaussian_filter

def create_heatmap_from_targets(
    targets,
    img_size=(512, 512),
    range_min=0,
    range_max=200,
    vel_min=-40,
    vel_max=40,
    sigma=15.0
):
    """
    生成热力图（ControlNet 条件输入）
    坐标系：上=近（0m），下=远（200m）；左=负速，右=正速
    """
    H, W = img_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    for r, v in targets:
        r_norm = np.clip((r - range_min) / (range_max - range_min), 0.0, 1.0)
        y = int(r_norm * (H - 1))
        v_norm = np.clip((v - vel_min) / (vel_max - vel_min), 0.0, 1.0)
        x = int(v_norm * (W - 1))
        
        # 为每个目标创建一个单独的高斯核
        single_point = np.zeros((H, W), dtype=np.float32)
        single_point[y, x] = 1.0
        gaussian_blob = gaussian_filter(single_point, sigma=sigma)
        
        # 归一化单个高斯核，然后叠加到热力图上
        if gaussian_blob.max() > 0:
            gaussian_blob = gaussian_blob / gaussian_blob.max()
        heatmap += gaussian_blob
    
    # 最后对整个热力图进行clip，确保值在合理范围内
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap
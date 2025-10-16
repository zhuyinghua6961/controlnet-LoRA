import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import argparse
import os
import re

def create_heatmap(targets, img_size=(512, 512), range_max=200, vel_max=40, sigma=15.0):
    """
    生成热力图控制图像
    targets: [(distance, velocity), ...] 例如 [(143, -20), (80, 5)]
    """
    H, W = img_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    for r, v in targets:
        # 距离归一化到Y轴 (0=近, 511=远)
        r_norm = np.clip(r / range_max, 0.0, 1.0)
        y = int(r_norm * (H - 1))
        
        # 速度归一化到X轴 (0=负速, 511=正速)
        v_norm = np.clip((v + vel_max) / (2 * vel_max), 0.0, 1.0)
        x = int(v_norm * (W - 1))
        
        # 创建高斯点
        single_point = np.zeros((H, W), dtype=np.float32)
        single_point[y, x] = 1.0
        gaussian_blob = gaussian_filter(single_point, sigma=sigma)
        
        if gaussian_blob.max() > 0:
            gaussian_blob = gaussian_blob / gaussian_blob.max()
        heatmap += gaussian_blob
    
    heatmap = np.clip(heatmap, 0.0, 1.0)
    
    # 转换为RGB图像
    heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1)
    image = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))
    return image

def parse_targets_string(targets_string):
    """
    解析目标字符串
    支持两种格式:
    1. 单行: targets=[(143,-20)],targets=[(144,30),(52,21)]
    2. 多行: 每行一个targets定义
       targets=[(143,-20)]
       targets=[(144,30),(52,21)]
    返回: [[(143,-20)], [(144,30),(52,21)]]
    """
    targets_list = []
    
    # 先尝试按换行符分割（多行格式）
    lines = targets_string.strip().split('\n')
    
    # 如果只有一行，则尝试按逗号分割（单行格式）
    if len(lines) == 1:
        parts = []
        current_part = ""
        bracket_level = 0
        
        for char in targets_string:
            if char == '[':
                bracket_level += 1
            elif char == ']':
                bracket_level -= 1
            
            if char == ',' and bracket_level == 0:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
    else:
        # 多行格式，每行就是一个part
        parts = [line.strip() for line in lines if line.strip()]
    
    # 解析每一部分
    for part in parts:
        # 使用正则表达式匹配 targets=[(...)]
        pattern = r'targets=\[((?:\([^)]+\)(?:,\s*)?)+)\]'
        match = re.search(pattern, part)
        
        if match:
            # 解析单个targets中的坐标对
            coords_pattern = r'\(([^)]+)\)'
            coords = re.findall(coords_pattern, match.group(1))
            
            target_group = []
            for coord in coords:
                # 分割坐标值
                values = coord.split(',')
                if len(values) == 2:
                    try:
                        distance = float(values[0].strip())
                        velocity = float(values[1].strip())
                        target_group.append((distance, velocity))
                    except ValueError:
                        continue
            
            if target_group:
                targets_list.append(target_group)
    
    return targets_list

def main():
    parser = argparse.ArgumentParser(description='批量生成雷达控制图像')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--targets_file', type=str, help='目标配置文件路径 (.txt)')
    parser.add_argument('--targets', type=str, help='目标配置字符串（直接输入）')
    parser.add_argument('--range_max', type=float, default=200, help='最大距离 (默认: 200)')
    parser.add_argument('--vel_max', type=float, default=40, help='最大速度 (默认: 40)')
    parser.add_argument('--sigma', type=float, default=15.0, help='高斯模糊sigma (默认: 15.0)')
    
    args = parser.parse_args()
    
    # 检查必须提供targets或targets_file之一
    if not args.targets and not args.targets_file:
        print("❌ 错误：必须提供 --targets 或 --targets_file 参数之一")
        parser.print_help()
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取目标配置字符串
    targets_string = ""
    if args.targets_file:
        # 从文件读取
        if not os.path.exists(args.targets_file):
            print(f"❌ 错误：文件不存在: {args.targets_file}")
            return
        
        print(f"📄 从文件读取配置: {args.targets_file}")
        with open(args.targets_file, 'r', encoding='utf-8') as f:
            targets_string = f.read().strip()
    else:
        # 从命令行参数读取
        targets_string = args.targets
    
    # 解析目标字符串
    try:
        targets_list = parse_targets_string(targets_string)
    except Exception as e:
        print(f"❌ 解析目标字符串失败: {e}")
        print(f"输入的字符串: {targets_string[:200]}...")
        return
    
    if not targets_list:
        print("❌ 未解析到任何目标配置")
        return
    
    print(f"📊 共解析到 {len(targets_list)} 个目标配置")
    print()
    
    # 批量生成图像
    for idx, targets in enumerate(targets_list, 1):
        print(f"生成第 {idx} 张图像...")
        print(f"  目标数量: {len(targets)}")
        print(f"  目标坐标: {targets}")
        
        # 生成热力图
        control_image = create_heatmap(
            targets,
            range_max=args.range_max,
            vel_max=args.vel_max,
            sigma=args.sigma
        )
        
        # 保存图像
        output_path = os.path.join(args.output_dir, f"control_image_{idx:03d}.png")
        control_image.save(output_path)
        print(f"  ✅ 已保存: {output_path}")
        print()
    
    print(f"🎉 完成！共生成 {len(targets_list)} 张控制图像")
    print(f"📁 输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()
"""
比较原始版本和改进版本的结果
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import sys

def run_original_code():
    """运行原始代码并保存结果"""
    print("="*60)
    print("运行原始版本代码...")
    print("="*60)

    # 动态导入原始代码
    sys.path.insert(0, '/mnt/c/Users/XiaofanGui/Theoretical_magic_angle')

    # 修改导入以避免重复运行
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "original",
        "/mnt/c/Users/XiaofanGui/Theoretical_magic_angle/Theoretical_magic_angle.py"
    )

    # 注意：原始代码可能会直接执行，所以我们需要小心处理
    print("注意：原始代码需要手动运行")
    return None

def load_results():
    """加载两个版本的结果"""
    try:
        # 检查是否存在保存的图像
        import os

        original_exists = os.path.exists('alpha_sample_map.png')
        improved_exists = os.path.exists('alpha_sample_map_improved.png')

        print(f"\n文件检查：")
        print(f"  原始版本图像: {'✓' if original_exists else '✗'}")
        print(f"  改进版本图像: {'✓' if improved_exists else '✗'}")

        return original_exists, improved_exists
    except Exception as e:
        print(f"加载结果时出错: {e}")
        return False, False

def create_comparison_plot():
    """创建对比可视化"""
    import os
    from PIL import Image

    print("\n创建对比图...")

    files = {
        'original': 'alpha_sample_map.png',
        'improved': 'alpha_sample_map_improved.png',
        'original_fine': 'alpha_sample_fine_scan.png',
        'improved_fine': 'alpha_sample_fine_scan_improved.png'
    }

    # 检查哪些文件存在
    available = {k: v for k, v in files.items() if os.path.exists(v)}

    if len(available) < 2:
        print("需要至少两个结果文件才能进行对比")
        return

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (key, filename) in enumerate(list(available.items())[:2]):
        try:
            img = Image.open(filename)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'{key.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        except Exception as e:
            print(f"无法加载 {filename}: {e}")

    plt.suptitle('原始版本 vs 改进版本对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_original_vs_improved.png', dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: comparison_original_vs_improved.png")

    return available

if __name__ == "__main__":
    print("="*60)
    print("代码对比工具")
    print("="*60)

    # 检查结果
    orig_exists, impr_exists = load_results()

    if orig_exists and impr_exists:
        print("\n两个版本的结果都已生成！")
        create_comparison_plot()
    elif impr_exists:
        print("\n改进版本已完成，等待原始版本...")
        print("请运行: python3 Theoretical_magic_angle.py")
    elif orig_exists:
        print("\n原始版本已完成，等待改进版本...")
        print("请运行: python3 Theoretical_magic_angle_improved.py")
    else:
        print("\n未找到任何结果文件")
        print("请先运行代码生成结果")

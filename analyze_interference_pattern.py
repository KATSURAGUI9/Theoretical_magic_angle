"""
分析干涉图样的形成 - 为什么是条纹而不是块状？
"""
import numpy as np

def analyze_pattern(theta_foil, theta_laser):
    """分析给定角度下的干涉图样特征"""

    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)

    # 入射波波矢方向
    k_in = np.array([np.sin(theta_l), np.cos(theta_l)])

    # 反射波波矢方向
    k_ref = np.array([np.sin(theta_l), -np.cos(theta_l)])

    # 薄膜法向
    n_foil = np.array([np.sin(theta_f), np.cos(theta_f)])

    print(f"\n{'='*70}")
    print(f"角度配置: θ_foil={theta_foil}°, θ_laser={theta_laser}°")
    print(f"{'='*70}")

    print(f"\n入射波波矢: k_in = [{k_in[0]:.4f}, {k_in[1]:.4f}]")
    print(f"反射波波矢: k_ref = [{k_ref[0]:.4f}, {k_ref[1]:.4f}]")
    print(f"薄膜法向:   n = [{n_foil[0]:.4f}, {n_foil[1]:.4f}]")

    # 驻波波矢
    k_standing = k_in - k_ref
    print(f"\n驻波波矢: Δk = k_in - k_ref = [{k_standing[0]:.4f}, {k_standing[1]:.4f}]")
    print(f"驻波方向: 垂直于 [{k_standing[1]:.4f}, {-k_standing[0]:.4f}]")

    # 驻波条纹的方向
    if abs(k_standing[0]) > 1e-6:
        angle_stripes = np.degrees(np.arctan(k_standing[1] / k_standing[0]))
        print(f"条纹角度: {angle_stripes:.2f}° (相对于x轴)")
    else:
        print(f"条纹方向: 水平")

    # 与薄膜的夹角
    dot_product = np.dot(k_standing, n_foil)
    angle_to_membrane = np.degrees(np.arccos(abs(dot_product) / np.linalg.norm(k_standing)))
    print(f"条纹与薄膜夹角: {angle_to_membrane:.2f}°")

    if angle_to_membrane < 10:
        print("⚠️  条纹几乎平行薄膜 → 可能产生块状图样（Moiré效应）")
    elif angle_to_membrane > 80:
        print("⚠️  条纹几乎垂直薄膜 → 清晰的线状条纹")
    else:
        print("✓  条纹与薄膜倾斜 → 斜向条纹")

    return angle_to_membrane

print("="*70)
print("干涉图样分析：为什么是条纹而不是块状？")
print("="*70)

# 测试不同的角度组合
configs = [
    (30, 65),  # 当前配置
    (0, 45),   # 论文可能配置1
    (45, 45),  # 论文可能配置2
    (60, 30),  # 论文可能配置3
]

angles_to_membrane = []
for theta_f, theta_l in configs:
    angle = analyze_pattern(theta_f, theta_l)
    angles_to_membrane.append((theta_f, theta_l, angle))

print("\n" + "="*70)
print("总结：产生块状图样的条件")
print("="*70)
print("\n块状图样（而不是线状条纹）的物理原因：")
print("1. 条纹方向与薄膜接近平行（夹角 < 15°）")
print("2. 产生摩尔条纹（Moiré pattern）效应")
print("3. 多次反射的相干叠加在薄膜附近形成局部增强")
print("\n推荐尝试的角度组合（可能产生块状）：")

for theta_f, theta_l, angle in angles_to_membrane:
    if angle < 15 or angle > 75:
        print(f"  ✓ θ_foil={theta_f}°, θ_laser={theta_l}° (夹角={angle:.1f}°)")
    else:
        print(f"    θ_foil={theta_f}°, θ_laser={theta_l}° (夹角={angle:.1f}°)")

print("\n" + "="*70)
print("💡 建议：尝试 θ_foil=0° 或 θ_foil=45° 看看是否出现块状图样")
print("="*70)

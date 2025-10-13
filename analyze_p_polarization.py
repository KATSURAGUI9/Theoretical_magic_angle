"""
分析 p 偏振光的 Ex 和 Ez 分量比例
"""
import numpy as np
import matplotlib.pyplot as plt

theta_laser = 65  # 度
theta_l = np.radians(theta_laser)

print("="*60)
print("p 偏振光的矢量分解")
print("="*60)
print(f"激光入射角: {theta_laser}°")
print()

# p 偏振的电场方向（在 xz 平面内，垂直于传播方向）
# 对于沿 (sin θ, cos θ) 传播的光，电场方向是 (cos θ, -sin θ)
Ex_coeff = np.cos(theta_l)
Ez_coeff = -np.sin(theta_l)

print(f"入射波电场方向（归一化）：")
print(f"  Ex = E₀ × {Ex_coeff:.6f}")
print(f"  Ez = E₀ × {Ez_coeff:.6f}")
print(f"  |E| = E₀ × {np.sqrt(Ex_coeff**2 + Ez_coeff**2):.6f}")
print()

print(f"分量比例：")
print(f"  |Ex| / |E| = {abs(Ex_coeff):.2%}")
print(f"  |Ez| / |E| = {abs(Ez_coeff):.2%}")
print()

print("="*60)
print("结论：")
print("="*60)
print(f"对于 {theta_laser}° 入射角：")
print(f"  - Ex 占总场强的 {abs(Ex_coeff):.1%}")
print(f"  - Ez 占总场强的 {abs(Ez_coeff):.1%}")
print(f"  - Ez 比 Ex 大 {abs(Ez_coeff/Ex_coeff):.2f} 倍！")
print()
print("⚠️  所以只显示 Ex 确实会丢失大量信息！")
print()

# 测试不同角度
print("="*60)
print("不同入射角下的 Ex/Ez 比例：")
print("="*60)
angles = [30, 45, 60, 65, 70, 80]
print(f"{'角度':<8} {'Ex/E₀':<10} {'Ez/E₀':<10} {'|Ex|/|Ez|':<12}")
print("-"*60)
for angle in angles:
    theta = np.radians(angle)
    ex = np.cos(theta)
    ez = -np.sin(theta)
    ratio = abs(ex/ez) if abs(ez) > 0.01 else np.inf
    print(f"{angle}°{'':<5} {ex:>8.4f}   {ez:>8.4f}   {ratio:>8.4f}")

print()
print("观察：")
print("  - 角度越大，Ez 占比越大")
print("  - 对于 65°，Ez 是主导分量（91% vs 42%）")
print("  - 只显示 Ex 会严重低估总场强")

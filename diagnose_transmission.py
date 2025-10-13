"""
诊断透射波消失的问题
"""
import numpy as np
import matplotlib.pyplot as plt

# 参数
theta_laser = 65  # 度
theta_l_rad = np.radians(theta_laser)

# 模拟透射区域的场（简化）
# 假设透射系数 t_total = 0.4
t_total = 0.4
E0 = 1.0

# 透射波的电场分量
Ex_trans = t_total * E0 * np.cos(theta_l_rad)
Ez_trans = -t_total * E0 * np.sin(theta_l_rad)

print("="*60)
print("透射波诊断")
print("="*60)
print(f"激光角度: {theta_laser}°")
print(f"透射系数 t_total: {t_total}")
print(f"\n透射波电场分量：")
print(f"  Ex_trans = {Ex_trans:.6f}")
print(f"  Ez_trans = {Ez_trans:.6f}")
print(f"  |E_trans| = {np.sqrt(Ex_trans**2 + Ez_trans**2):.6f}")

# 测试不同的显示方案
print(f"\n不同显示方案的值：")
print("-"*60)

# 方案1：直接显示 Ex
E_display_1 = Ex_trans
print(f"1. Ex:              {E_display_1:.6f}")

# 方案2：显示 E_perp（当前方案）
E_perp = Ex_trans * np.sin(theta_l_rad) + Ez_trans * np.cos(theta_l_rad)
print(f"2. E_perp:          {E_perp:.6f}")

# 方案3：显示 |E|
E_mag = np.sqrt(Ex_trans**2 + Ez_trans**2)
print(f"3. |E|:             {E_mag:.6f}")

# 方案4：显示 sign(Ex) * |E|
E_signed = np.sign(Ex_trans) * E_mag
print(f"4. sign(Ex) × |E|:  {E_signed:.6f}")

print("\n" + "="*60)
print("问题分析：")
print("="*60)

# 理论上，p偏振光的横波分量应该等于总场强
# 因为 E ⊥ k（横波条件）
print(f"\n理论验证：")
print(f"  E·k = Ex*kx + Ez*kz")
print(f"      = {Ex_trans:.6f} × {np.sin(theta_l_rad):.6f} + {Ez_trans:.6f} × {np.cos(theta_l_rad):.6f}")
print(f"      = {Ex_trans * np.sin(theta_l_rad) + Ez_trans * np.cos(theta_l_rad):.6f}")
print(f"  (应该 ≈ 0，因为 E ⊥ k)")

print(f"\n✓ 如果 E·k ≈ 0，说明 E_perp 应该等于 |E|")
print(f"  实际：E_perp = {E_perp:.6f}, |E| = {E_mag:.6f}")

if abs(E_perp) < 0.01 * E_mag:
    print(f"\n❌ 问题确认：E_perp ≈ 0，导致透射波视觉上消失！")
    print(f"   这是因为坐标变换公式不对。")
    print(f"\n正确的做法：")
    print(f"  - 入射侧：E_perp 是垂直于激光传播方向的分量")
    print(f"  - 透射侧：应该直接显示 |E| 或 Ex（主导分量）")
else:
    print(f"\n✓ E_perp 不为零，透射波应该可见")

print("\n" + "="*60)
print("推荐方案：")
print("="*60)
print("方案A：直接显示 Ex（最简单，透射波清晰可见）")
print("  E_display = Ex_norm")
print("\n方案B：显示 sign(Ex) × |E|（保留符号，振幅清晰）")
print("  E_display = np.sign(Ex_norm) * E_magnitude_norm")
print("\n方案C：使用不同的坐标变换（需要区分入射/反射/透射）")

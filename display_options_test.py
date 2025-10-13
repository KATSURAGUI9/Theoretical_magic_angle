"""
测试不同的电场显示方案，找到最接近论文的效果
"""
import numpy as np
import matplotlib.pyplot as plt

# 模拟一些测试数据
x = np.linspace(-3e-6, 3e-6, 200)
z = np.linspace(-3e-6, 3e-6, 200)
X, Z = np.meshgrid(x, z)

# 模拟p偏振光场（入射+反射）
theta_l = np.radians(65)
k0 = 2 * np.pi / 1030e-9
omega = k0 * 3e8

kx = k0 * np.sin(theta_l)
kz = k0 * np.cos(theta_l)

# 入射波
phase_in = kx*X + kz*Z
Ex_in = np.cos(theta_l) * np.cos(phase_in)
Ez_in = -np.sin(theta_l) * np.cos(phase_in)

# 反射波
r = 0.5
phase_ref = kx*X - kz*Z
Ex_ref = r * np.cos(theta_l) * np.cos(phase_ref)
Ez_ref = r * np.sin(theta_l) * np.cos(phase_ref)

# 总场
Ex = Ex_in + Ex_ref
Ez = Ez_in + Ez_ref
E_mag = np.sqrt(Ex**2 + Ez**2)

# ============================================================================
# 测试5种显示方案
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 方案1：仅 Ex
ax = axes[0, 0]
im1 = ax.contourf(Z*1e6, X*1e6, Ex, levels=51, cmap='RdBu_r')
ax.set_title('方案1: Ex 分量', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im1, ax=ax, label='Ex')
ax.set_aspect('equal')

# 方案2：E_perp（垂直传播方向）
ax = axes[0, 1]
E_perp = Ex * np.sin(theta_l) + Ez * np.cos(theta_l)
im2 = ax.contourf(Z*1e6, X*1e6, E_perp, levels=51, cmap='RdBu_r')
ax.set_title('方案2: E_perp (垂直传播)', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im2, ax=ax, label='E_perp')
ax.set_aspect('equal')

# 方案3：带符号的场强（原方案）
ax = axes[0, 2]
E_signed = np.sign(Ex) * E_mag
im3 = ax.contourf(Z*1e6, X*1e6, E_signed, levels=51, cmap='RdBu_r')
ax.set_title('方案3: sign(Ex) × |E| (原方案)', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im3, ax=ax, label='signed |E|')
ax.set_aspect('equal')

# 方案4：相位加权
ax = axes[1, 0]
phase = np.arctan2(Ez, Ex)
E_phase = E_mag * np.cos(phase)
im4 = ax.contourf(Z*1e6, X*1e6, E_phase, levels=51, cmap='RdBu_r')
ax.set_title('方案4: |E|×cos(phase)', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im4, ax=ax, label='phase-weighted')
ax.set_aspect('equal')

# 方案5：场强大小（参考）
ax = axes[1, 1]
im5 = ax.contourf(Z*1e6, X*1e6, E_mag, levels=31, cmap='hot')
ax.set_title('方案5: |E| 场强', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im5, ax=ax, label='|E|')
ax.set_aspect('equal')

# 方案6：Ex 和 Ez 的加权平均
ax = axes[1, 2]
E_weighted = 0.8 * Ex + 0.2 * Ez
im6 = ax.contourf(Z*1e6, X*1e6, E_weighted, levels=51, cmap='RdBu_r')
ax.set_title('方案6: 0.8Ex + 0.2Ez', fontsize=14, fontweight='bold')
ax.set_xlabel('z (μm)')
ax.set_ylabel('x (μm)')
plt.colorbar(im6, ax=ax, label='weighted')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('display_options_comparison.png', dpi=300, bbox_inches='tight')
print("已保存对比图: display_options_comparison.png")
print("\n建议：")
print("- 如果论文显示主要是条纹：使用方案1（Ex）")
print("- 如果论文强调横波性质：使用方案2（E_perp）推荐✓")
print("- 如果论文有块状图样：使用方案4（相位加权）")
plt.show()

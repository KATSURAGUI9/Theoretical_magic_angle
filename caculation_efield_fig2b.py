import numpy as np
import matplotlib.pyplot as plt

# 物理参数
c = 299792458.0
WAVELENGTH = 1030e-9
E0 = 1e9
n = 3.6
d = 60e-9

# 关键配置：让薄膜接近垂直，激光斜入射
theta_foil = 35    # 薄膜法向几乎水平 → 薄膜接近垂直
theta_laser = 65   # 激光从左下斜入射

def fresnel_p(theta_in, n):
    """p偏振Fresnel系数"""
    cos_i = np.cos(theta_in)
    sin_i = np.sin(theta_in)
    sin_t = sin_i / n
    
    if abs(sin_t) > 1:
        return -1.0, 0.0
    
    cos_t = np.sqrt(1 - sin_t**2)
    r = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    t = 2 * cos_i / (n * cos_i + cos_t)
    
    return r, t

def calc_field_final(X, Z):
    """
    最终版本：确保右侧无波节
    """
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)
    
    omega = 2 * np.pi * c / WAVELENGTH
    k0 = omega / c
    
    # 入射波矢
    kx_in = k0 * np.sin(theta_l)
    kz_in = k0 * np.cos(theta_l)
    k_in = np.array([kx_in, kz_in])
    
    # 薄膜法向（theta_foil大时接近水平）
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)
    n_foil = np.array([nx, nz])
    
    # 计算入射角
    cos_theta_in = abs(np.dot(k_in, n_foil)) / k0
    theta_in = np.arccos(np.clip(cos_theta_in, 0, 1))
    
    # Fresnel系数（含多次反射）
    r12, t12 = fresnel_p(theta_in, n)
    
    sin_t = np.sin(theta_in) / n
    cos_t = np.sqrt(1 - sin_t**2)
    theta_t = np.arcsin(sin_t)
    
    r21, t21 = fresnel_p(theta_t, 1/n)
    
    # 多次反射总系数
    phi = 2 * n * k0 * d * cos_t
    denom = 1 - r21**2 * np.exp(1j * phi)
    r_eff = r12 + t12 * t21 * r21 * np.exp(1j * phi) / denom
    t_eff = t12 * t21 * np.exp(1j * phi/2) / denom
    
    # 点到薄膜的距离（薄膜前表面在原点）
    dist = X * nx + Z * nz
    
    # 严格区域划分
    left = (dist < -1e-10)          # 左侧：入射+反射
    right = (dist > d + 1e-10)      # 右侧：仅透射
    membrane = ~left & ~right       # 薄膜
    
    # 初始化
    E = np.zeros_like(X, dtype=complex)
    
    # ==========================================
    # 左侧：入射波 + 反射波（驻波）
    # ==========================================
    # 入射波相位
    phase_in = kx_in * X + kz_in * Z
    E_in = E0 * np.exp(1j * phase_in)
    
    # 反射波矢（关于法线镜像 - 这是关键！）
    k_dot_n = np.dot(k_in, n_foil)
    k_ref = k_in - 2 * k_dot_n * n_foil
    kx_ref = k_ref[0]
    kz_ref = k_ref[1]
    
    phase_ref = kx_ref * X + kz_ref * Z
    E_ref = r_eff * E0 * np.exp(1j * phase_ref)
    
    E[left] = E_in[left] + E_ref[left]
    
    # ==========================================
    # 薄膜内：折射场（考虑多次反射）
    # ==========================================
    # 在薄膜内，需要使用折射后的波矢
    # 折射定律：平行于界面的分量保持不变
    # k_parallel = k0 * sin(theta_in)
    # 在薄膜内：k_perp = n*k0*cos(theta_t)

    # 方法：直接用入射波矢和折射率计算薄膜内的场
    # 注意这是简化，实际应该考虑折射后的传播方向
    k_membrane = n * k0
    phase_mem = k_membrane * (kx_in/k0 * X + kz_in/k0 * Z)

    # 薄膜内有入射和多次反射的叠加，这里用有效透射系数
    E[membrane] = t12 * E0 * np.exp(1j * phase_mem[membrane])

    # ==========================================
    # 右侧：仅透射波（行波 - 不应有波节！）
    # ==========================================
    # 关键：透射波应该是纯行波，相位连续
    # 考虑通过薄膜的总相位变化
    phase_trans = kx_in * X + kz_in * Z
    E[right] = t_eff * E0 * np.exp(1j * phase_trans[right])
    
    return np.real(E), left, membrane, right


# 生成网格 (修正：确保坐标系一致)
x = np.linspace(-2e-6, 2e-6, 1000)
z = np.linspace(-2e-6, 2e-6, 1000)
Z, X = np.meshgrid(z, x)  # 注意：Z对应横轴，X对应纵轴

print(f"计算场分布（薄膜倾角={theta_foil}°）...")
E_field, m_left, m_mem, m_right = calc_field_final(X, Z)

# 归一化
E_norm = E_field / E0

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# 左图：电场分布
ax = axes[0]
im = ax.imshow(E_norm, extent=[-2, 2, -2, 2],
               cmap='RdBu_r', vmin=-1.5, vmax=1.5,
               aspect='equal', origin='lower', 
               interpolation='bilinear')

# 画薄膜位置
theta_f_rad = np.radians(theta_foil)
z_line = np.linspace(-3, 3, 100)
x_line = -z_line * np.cos(theta_f_rad) / np.sin(theta_f_rad)
ax.plot(z_line, x_line, 'k--', linewidth=3, alpha=0.9)

# 激光方向箭头
theta_l_rad = np.radians(theta_laser)
ax.arrow(-1.5, -1.5, 
         0.5*np.sin(theta_l_rad), 0.5*np.cos(theta_l_rad),
         head_width=0.12, head_length=0.08, 
         fc='red', ec='red', linewidth=2, zorder=10)
ax.text(-1.6, -1.7, 'Laser', fontsize=12, color='red', fontweight='bold')

ax.set_xlabel('z (μm)', fontsize=14, fontweight='bold')
ax.set_ylabel('x (μm)', fontsize=14, fontweight='bold')
ax.set_title(f'Electric Field (θ_foil={theta_foil}°, θ_laser={theta_laser}°)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.2)

cbar = plt.colorbar(im, ax=ax, label='E / E₀', fraction=0.046, pad=0.04)

# 右图：区域验证
ax = axes[1]
region = np.zeros_like(X)
region[m_left] = 1
region[m_mem] = 2
region[m_right] = 3

from matplotlib.colors import ListedColormap
cmap_reg = ListedColormap(['white', 'cyan', 'gray', 'yellow'])
im2 = ax.contourf(Z*1e6, X*1e6, region,
                  levels=[0, 0.5, 1.5, 2.5, 3.5],
                  cmap=cmap_reg)

ax.plot(z_line, x_line, 'k--', linewidth=3)

ax.text(-1.3, 0, 'LEFT\n(Standing\nWave)', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7),
        ha='center', fontweight='bold')
ax.text(1.3, 0, 'RIGHT\n(Traveling\nWave)', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        ha='center', fontweight='bold')

ax.set_xlabel('z (μm)', fontsize=14)
ax.set_ylabel('x (μm)', fontsize=14)
ax.set_title('Region Verification', fontsize=14, fontweight='bold')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('field_final_correct.png', dpi=300, bbox_inches='tight')
plt.show()

# 验证右侧是否有零点
right_field = E_norm[m_right]
n_zeros = np.sum(np.abs(right_field) < 0.05)
print(f"\n✓ 区域统计:")
print(f"  左侧: {np.sum(m_left):,} 像素")
print(f"  薄膜: {np.sum(m_mem):,} 像素")
print(f"  右侧: {np.sum(m_right):,} 像素")
print(f"\n验证右侧（透射区）:")
print(f"  场强范围: [{right_field.min():.3f}, {right_field.max():.3f}]")
print(f"  接近零的点: {n_zeros} ({100*n_zeros/np.sum(m_right):.2f}%)")
print(f"  → 如果这个比例很小(<5%)，说明右侧没有波节 ✓")
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 物理常数和参数
# ============================================================================
c = 299792458.0
WAVELENGTH = 1030e-9
E0 = 1e9
REFRACTIVE_INDEX = 3.6
MEMBRANE_THICKNESS = 60e-9
theta_foil = 30   # ✓ 修复：使用θ_foil=0°让条纹平行薄膜，产生块状图样
theta_laser = 45
t_snapshot = 0  # ✓ 修复：使用t=0看静态干涉，而不是300fs（87个周期）

# Fresnel系数
def fresnel_coefficients(theta_in, n, polarization='p'):
    cos_theta_in = np.cos(theta_in)
    sin_theta_in = np.sin(theta_in)
    sin_theta_t = sin_theta_in / n
    if np.abs(sin_theta_t) > 1:
        return -1.0, 0.0, 1.0, 0.0
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    if polarization == 'p':
        r = (n * cos_theta_in - cos_theta_t) / (n * cos_theta_in + cos_theta_t)
        t = 2 * cos_theta_in / (n * cos_theta_in + cos_theta_t)
        R = np.abs(r)**2
        T = np.abs(t)**2 * (n * cos_theta_t) / cos_theta_in
    else:
        r = (cos_theta_in - n * cos_theta_t) / (cos_theta_in + n * cos_theta_t)
        t = 2 * cos_theta_in / (cos_theta_in + n * cos_theta_t)
        R = np.abs(r)**2
        T = np.abs(t)**2 * (n * cos_theta_t) / cos_theta_in
    return r, t, R, T

# 多次反射
def calculate_multiple_reflections(r12, t12, r21, t21, k_film, d, max_order=5):
    phi = 2 * k_film * d
    phase_factor = np.exp(1j * phi)
    r_total = r12
    t_total = 0.0
    for m in range(max_order):
        internal_bounces = (r21**2)**m * phase_factor**(m+1)
        r_total += t12 * r21 * t21 * internal_bounces / phase_factor
        t_total += t12 * t21 * internal_bounces
    return r_total, t_total

# ============================================================================
# 【完全修复】电场计算 - 正确的区域划分
# ============================================================================
def calculate_em_field_fixed(x, z, t, theta_foil, theta_laser,
                              wavelength, E0, n, d, max_order=5):
    """
    完全修复版 - 正确的区域划分和物理
    """
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)
    
    omega = 2 * np.pi * c / wavelength
    k0 = omega / c
    
    # 入射波波矢
    k_in = k0 * np.array([np.sin(theta_l), np.cos(theta_l)])
    
    # 薄膜法向
    n_foil = np.array([np.sin(theta_f), np.cos(theta_f)])
    
    # 计算入射角
    cos_theta_in = np.abs(np.dot(k_in/k0, n_foil))
    theta_in = np.arccos(cos_theta_in)
    
    # Fresnel系数
    r12, t12, R12, T12 = fresnel_coefficients(theta_in, n, 'p')
    theta_t = np.arcsin(np.sin(theta_in) / n)
    r21, t21, R21, T21 = fresnel_coefficients(theta_t, 1.0/n, 'p')
    k_film = n * k0
    r_total, t_total = calculate_multiple_reflections(
        r12, t12, r21, t21, k_film * np.cos(theta_t), d, max_order
    )
    
    # 【关键修复】点到薄膜的有符号距离
    # 薄膜前表面在原点，法向为n_foil
    dist_to_membrane = x * n_foil[0] + z * n_foil[1]
    
    # 【正确的区域划分】
    mask_incident = dist_to_membrane < 0
    mask_inside = (dist_to_membrane >= 0) & (dist_to_membrane <= d)
    mask_transmitted = dist_to_membrane > d
    
    # 初始化场
    Ex = np.zeros_like(z, dtype=complex)
    Ez = np.zeros_like(z, dtype=complex)
    
    # 区域1：入射侧（入射波 + 反射波）
    if np.any(mask_incident):
        E_in_x = E0 * np.cos(theta_l)
        E_in_z = -E0 * np.sin(theta_l)
        phase_in = k_in[0]*x + k_in[1]*z - omega*t
        
        k_ref = k0 * np.array([np.sin(theta_l), -np.cos(theta_l)])
        E_ref_x = r_total * E0 * np.cos(theta_l)
        E_ref_z = r_total * E0 * np.sin(theta_l)
        phase_ref = k_ref[0]*x + k_ref[1]*z - omega*t
        
        Ex[mask_incident] = (E_in_x * np.exp(1j*phase_in[mask_incident]) +
                             E_ref_x * np.exp(1j*phase_ref[mask_incident]))
        Ez[mask_incident] = (E_in_z * np.exp(1j*phase_in[mask_incident]) +
                             E_ref_z * np.exp(1j*phase_ref[mask_incident]))
    
    # 区域2：薄膜内
    if np.any(mask_inside):
        phase_inside = k_in[0]*x + k_in[1]*z - omega*t
        Ex[mask_inside] = t12 * E0 * np.cos(theta_l) * np.exp(1j*phase_inside[mask_inside]) / n
        Ez[mask_inside] = -t12 * E0 * np.sin(theta_l) * np.exp(1j*phase_inside[mask_inside]) / n
    
    # 区域3：透射侧（只有透射波）
    if np.any(mask_transmitted):
        E_trans_x = t_total * E0 * np.cos(theta_l)
        E_trans_z = -t_total * E0 * np.sin(theta_l)
        
        phi_delay = n * k0 * d * np.cos(theta_t) + np.angle(t_total)
        phase_trans = k_in[0]*x + k_in[1]*z - omega*t + phi_delay
        
        Ex[mask_transmitted] = E_trans_x * np.exp(1j*phase_trans[mask_transmitted])
        Ez[mask_transmitted] = E_trans_z * np.exp(1j*phase_trans[mask_transmitted])
    
    return Ex.real, Ez.real, mask_incident, mask_inside, mask_transmitted


# ============================================================================
# 可视化 - 干净清晰的版本
# ============================================================================
def plot_final_clean():
    """清理后的可视化 - 去掉误导性的装饰线"""
    
    x_range = np.linspace(-3e-6, 3e-6, 400)
    z_range = np.linspace(-3e-6, 3e-6, 400)
    X, Z = np.meshgrid(x_range, z_range)
    
    print("计算电场（完全修复+清理版）...")
    Ex, Ez, mask_inc, mask_in, mask_trans = calculate_em_field_fixed(
        X, Z, t_snapshot, theta_foil, theta_laser,
        WAVELENGTH, E0, REFRACTIVE_INDEX, MEMBRANE_THICKNESS, max_order=5
    )
    
    E_magnitude = np.sqrt(Ex**2 + Ez**2)
    Ex_norm = Ex / E0
    Ez_norm = Ez / E0
    E_magnitude_norm = E_magnitude / E0
    E_display = np.sign(Ex_norm) * E_magnitude_norm
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # ------------------------------------------------------------------------
    # 子图1：归一化电场
    # ------------------------------------------------------------------------
    ax = axes[0]
    levels = np.linspace(-1, 1, 51)
    im1 = ax.contourf(Z*1e6, X*1e6, E_display,
                      levels=levels, cmap='RdBu_r', extend='both')
    
    # 激光箭头标注
    arrow_start_z, arrow_start_x = -2.3e-6, -2.3e-6
    theta_l_rad = np.radians(theta_laser)
    arrow_length = 1.2e-6
    arrow_dz = arrow_length * np.cos(theta_l_rad)
    arrow_dx = arrow_length * np.sin(theta_l_rad)
    ax.arrow(arrow_start_z*1e6, arrow_start_x*1e6,
             arrow_dz*1e6, arrow_dx*1e6,
             head_width=0.2, head_length=0.15, fc='red', ec='red',
             linewidth=2.5, zorder=15)
    ax.text(arrow_start_z*1e6-0.5, arrow_start_x*1e6-0.6,
            'Incident', fontsize=12, color='red', fontweight='bold')
    
    ax.set_xlabel('z (μm)', fontsize=13)
    ax.set_ylabel('x (μm)', fontsize=13)
    ax.set_title('Electric Field Distribution', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('E / E₀', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # ------------------------------------------------------------------------
    # 子图2：区域标记（物理区域）
    # ------------------------------------------------------------------------
    ax = axes[1]
    
    region_map = np.zeros_like(X)
    region_map[mask_inc] = 1
    region_map[mask_in] = 2
    region_map[mask_trans] = 3
    
    from matplotlib.colors import ListedColormap
    cmap_regions = ListedColormap(['white', 'cyan', 'gray', 'yellow'])
    im2 = ax.contourf(Z*1e6, X*1e6, region_map, 
                      levels=[0, 0.5, 1.5, 2.5, 3.5],
                      cmap=cmap_regions)
    
    # 文字标注（不画边框线）
    ax.text(-2, -2, 'Incident\n(Standing Wave)', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8),
            ha='center', fontweight='bold')
    ax.text(2, 2, 'Transmitted\n(Traveling Wave)', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            ha='center', fontweight='bold')
    ax.text(0, 0, 'Membrane\n(60nm)', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
            ha='center', fontweight='bold')
    
    ax.set_xlabel('z (μm)', fontsize=13)
    ax.set_ylabel('x (μm)', fontsize=13)
    ax.set_title('Physical Regions (No Decoration)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # ------------------------------------------------------------------------
    # 子图3：场强大小
    # ------------------------------------------------------------------------
    ax = axes[2]
    levels_mag = np.linspace(0, 1.5, 31)
    im3 = ax.contourf(Z*1e6, X*1e6, E_magnitude_norm,
                      levels=levels_mag, cmap='hot', extend='max')
    
    ax.set_xlabel('z (μm)', fontsize=13)
    ax.set_ylabel('x (μm)', fontsize=13)
    ax.set_title('Field Magnitude |E|/E₀', fontsize=14, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cbar3.set_label('|E| / E₀', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('field_CLEAN_VERSION.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: field_CLEAN_VERSION.png")
    plt.show()
    
    # 统计
    print(f"\n📊 区域统计：")
    print(f"  入射侧: {np.sum(mask_inc):,} 像素")
    print(f"  薄膜内: {np.sum(mask_in):,} 像素")
    print(f"  透射侧: {np.sum(mask_trans):,} 像素")
    total_pixels = mask_inc.size
    print(f"  总计: {total_pixels:,} 像素")
    print(f"  入射侧占比: {100*np.sum(mask_inc)/total_pixels:.1f}%")
    print(f"  透射侧占比: {100*np.sum(mask_trans)/total_pixels:.1f}%")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("清理版 - 去掉误导性装饰，只显示物理真实")
    print("="*70)
    print(f"参数: λ={WAVELENGTH*1e9:.0f}nm, E₀={E0*1e-9:.1f}GV/m, " +
          f"n={REFRACTIVE_INDEX}, d={MEMBRANE_THICKNESS*1e9:.0f}nm")
    print(f"角度: θ_foil={theta_foil}°, θ_laser={theta_laser}°")
    print(f"时间快照: t={t_snapshot*1e15:.2f}fs")
    print("="*70)
    
    plot_final_clean()
    
    print("\n" + "="*70)
    print("✅ 关键改进:")
    print("  1. 去掉了误导性的黑色Polygon边框")
    print("  2. 灰色区域 = 物理计算的真实薄膜位置")
    print("  3. 区域划分纯粹基于点到平面距离")
    print("="*70)
    print("\n💡 现在图像干净清晰，没有混淆！")
    print("="*70)
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 物理常数
# ============================================================================
c = 299792458.0           # 光速 m/s

# ============================================================================
# 论文参数（Figure 2b）
# ============================================================================
WAVELENGTH = 1030e-9      # 1030 nm
E0 = 1e9                  # 1 GV/m
REFRACTIVE_INDEX = 3.6    # Si
MEMBRANE_THICKNESS = 60e-9  # 60 nm

# 角度（论文Figure 2b）
theta_foil = 30           # 度
theta_laser = 65          # 度

# 时间快照（可以调整来看不同相位）
t_snapshot = 0            # 时间 = 0

# ============================================================================
# Fresnel 系数
# ============================================================================
def fresnel_coefficients(theta_in, n, polarization='p'):
    """计算 Fresnel 反射和透射系数"""
    cos_theta_in = np.cos(theta_in)
    sin_theta_in = np.sin(theta_in)
    
    # Snell 定律
    sin_theta_t = sin_theta_in / n
    
    # 检查全反射
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


# ============================================================================
# 多次反射求和
# ============================================================================
def calculate_multiple_reflections(r12, t12, r21, t21, k_film, d, max_order=5):
    """计算薄膜多次反射的累积效应"""
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
# 电磁场计算（简化版，只计算薄膜前后）
# ============================================================================
def calculate_em_field_2d(x, z, t, theta_foil, theta_laser, 
                          wavelength, E0, n, d, max_order=5):
    """
    计算2D空间中的电磁场
    
    参数:
        x, z: 空间坐标（可以是标量或数组）
        t: 时间
        其他参数同原代码
    
    返回:
        Ex, Ez: 电场的x和z分量
    """
    # 转换为弧度
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)
    
    cos_theta_f = np.cos(theta_f)
    sin_theta_f = np.sin(theta_f)
    cos_theta_l = np.cos(theta_l)
    sin_theta_l = np.sin(theta_l)
    
    omega = 2 * np.pi * c / wavelength
    k0 = omega / c
    
    # 入射波波矢
    k_in = k0 * np.array([sin_theta_l, 0, cos_theta_l])
    
    # 薄膜法向
    n_foil = np.array([sin_theta_f, 0, cos_theta_f])
    
    # 入射角（相对于薄膜法向）
    theta_in = np.arccos(np.abs(np.dot(k_in/k0, n_foil)))
    
    # Fresnel 系数
    r12, t12, R12, T12 = fresnel_coefficients(theta_in, n, 'p')
    theta_t = np.arcsin(np.sin(theta_in) / n)
    r21, t21, R21, T21 = fresnel_coefficients(theta_t, 1.0/n, 'p')
    
    # 薄膜中的波矢大小
    k_film = n * k0 * np.cos(theta_t)
    
    # 多次反射总系数
    r_total, t_total = calculate_multiple_reflections(
        r12, t12, r21, t21, k_film, d, max_order
    )
    
    # 薄膜在原点，法向沿着 n_foil
    # 计算点到薄膜的有符号距离
    # 简化：假设薄膜在z=0，倾斜角为theta_foil
    z_rotated = z * cos_theta_f + x * sin_theta_f
    
    # 判断点在薄膜前还是后
    # z_rotated < -d/2: 薄膜前（入射侧）
    # |z_rotated| <= d/2: 薄膜内
    # z_rotated > d/2: 薄膜后（透射侧）
    
    # 初始化电场
    Ex = np.zeros_like(z, dtype=complex)
    Ez = np.zeros_like(z, dtype=complex)
    
    # 入射波电场振幅（p偏振）
    E_in_amp_x = E0 * cos_theta_l
    E_in_amp_z = -E0 * sin_theta_l
    
    # 反射波电场振幅
    E_ref_amp_x = r_total * E0 * cos_theta_l
    E_ref_amp_z = r_total * E0 * sin_theta_l  # 注意符号变化
    
    # 透射波电场振幅
    E_trans_amp_x = t_total * E0 * cos_theta_l
    E_trans_amp_z = -t_total * E0 * sin_theta_l
    
    # 反射波波矢
    k_ref = k0 * np.array([sin_theta_l, 0, -cos_theta_l])
    
    # 计算每个点的相位
    r_vec = np.array([x, 0*x, z])
    
    # 入射波相位
    phase_in = k_in[0]*x + k_in[2]*z - omega*t
    
    # 反射波相位
    phase_ref = k_ref[0]*x + k_ref[2]*z - omega*t
    
    # 透射波相位（需要考虑通过薄膜的相位延迟）
    phase_trans = k_in[0]*x + k_in[2]*z - omega*t + np.angle(t_total)
    
    # 根据位置分配场
    mask_before = z_rotated < -d/2
    mask_after = z_rotated > d/2
    
    # 薄膜前：入射波 + 反射波
    Ex[mask_before] = (E_in_amp_x * np.exp(1j*phase_in[mask_before]) + 
                       E_ref_amp_x * np.exp(1j*phase_ref[mask_before]))
    Ez[mask_before] = (E_in_amp_z * np.exp(1j*phase_in[mask_before]) + 
                       E_ref_amp_z * np.exp(1j*phase_ref[mask_before]))
    
    # 薄膜后：透射波
    Ex[mask_after] = E_trans_amp_x * np.exp(1j*phase_trans[mask_after])
    Ez[mask_after] = E_trans_amp_z * np.exp(1j*phase_trans[mask_after])
    
    # 薄膜内：简化处理（可以更精细）
    mask_inside = ~mask_before & ~mask_after
    Ex[mask_inside] = t12 * E0 * cos_theta_l * np.exp(1j*phase_in[mask_inside]) / n
    Ez[mask_inside] = -t12 * E0 * sin_theta_l * np.exp(1j*phase_in[mask_inside]) / n
    
    # 返回实部（电场的瞬时值）
    return Ex.real, Ez.real


# ============================================================================
# 绘制电场分布图（类似Figure 2b）
# ============================================================================
def plot_field_distribution():
    """绘制电场分布，复现论文Figure 2(b)"""
    
    # 创建2D网格
    x_range = np.linspace(-2e-6, 2e-6, 200)  # -2 to 2 μm
    z_range = np.linspace(-2e-6, 2e-6, 200)  # -2 to 2 μm
    X, Z = np.meshgrid(x_range, z_range)
    
    print("计算电场分布...")
    # 计算电场
    Ex, Ez = calculate_em_field_2d(
        X, Z, t_snapshot,
        theta_foil, theta_laser,
        WAVELENGTH, E0, REFRACTIVE_INDEX, 
        MEMBRANE_THICKNESS, max_order=5
    )
    
    # 计算电场强度（可以选择不同的量来显示）
    E_magnitude = np.sqrt(Ex**2 + Ez**2)
    
    # 也可以只看x分量（通常是主导分量）
    E_display = Ex
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制电场（使用与论文相同的色图）
    levels = np.linspace(-E0, E0, 41)
    im = ax.contourf(Z*1e6, X*1e6, E_display, 
                     levels=levels, cmap='RdBu_r', extend='both')
    
    # 标注薄膜位置（倾斜的矩形）
    theta_f_rad = np.radians(theta_foil)
    membrane_x = np.array([-2, 2]) * 1e-6  # μm
    membrane_z_center = 0
    membrane_z = membrane_z_center + membrane_x * np.tan(theta_f_rad)
    
    # 画薄膜边界
    ax.plot(membrane_z*1e6, membrane_x*1e6, 'k-', linewidth=3, label='Membrane')
    ax.fill_between(membrane_z*1e6, membrane_x*1e6-MEMBRANE_THICKNESS*1e6/2, 
                     membrane_x*1e6+MEMBRANE_THICKNESS*1e6/2, 
                     color='gray', alpha=0.3)
    
    # 标注相位延迟线（波峰位置）
    # 找到入射侧和透射侧的波峰
    omega = 2 * np.pi * c / WAVELENGTH
    k0 = omega / c
    wavelength_display = WAVELENGTH
    
    # 画几条等相位线来显示相位延迟
    for i in range(-3, 4):
        phase_line = i * wavelength_display
        theta_l_rad = np.radians(theta_laser)
        z_line = phase_line * np.cos(theta_l_rad)
        x_line_range = np.array([-2, 2]) * 1e-6
        ax.plot([z_line*1e6, z_line*1e6], 
                [x_line_range[0]*1e6, x_line_range[1]*1e6], 
                'k--', linewidth=0.5, alpha=0.3)
    
    # 设置坐标轴
    ax.set_xlabel('z (μm)', fontsize=14)
    ax.set_ylabel('x (μm)', fontsize=14)
    ax.set_title(f'Electric Field Distribution (t={t_snapshot*1e15:.0f} as)\n' + 
                 f'θ_foil={theta_foil}°, θ_laser={theta_laser}°, Si 60nm',
                 fontsize=14)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Electric Field Ex (V/m)', fontsize=12)
    
    # 设置等比例
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 标注激光方向
    arrow_start_z = -1.5e-6
    arrow_start_x = -1.5e-6
    theta_l_rad = np.radians(theta_laser)
    arrow_length = 0.8e-6
    arrow_dz = arrow_length * np.cos(theta_l_rad)
    arrow_dx = arrow_length * np.sin(theta_l_rad)
    ax.arrow(arrow_start_z*1e6, arrow_start_x*1e6, 
             arrow_dz*1e6, arrow_dx*1e6,
             head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.text(arrow_start_z*1e6-0.3, arrow_start_x*1e6-0.3, 
            'Laser', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('field_distribution_PRA_Fig2b.png', dpi=300, bbox_inches='tight')
    print(f"已保存图像: field_distribution_PRA_Fig2b.png")
    plt.show()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("复现 PRA 论文 Figure 2(b)")
    print(f"参数:")
    print(f"  波长: {WAVELENGTH*1e9:.0f} nm")
    print(f"  电场: {E0*1e-9:.1f} GV/m")
    print(f"  折射率: {REFRACTIVE_INDEX}")
    print(f"  厚度: {MEMBRANE_THICKNESS*1e9:.0f} nm")
    print(f"  θ_foil: {theta_foil}°")
    print(f"  θ_laser: {theta_laser}°")
    print("="*60)
    
    plot_field_distribution()
    
    print("\n提示:")
    print("1. 可以修改 theta_foil 和 theta_laser 来看不同配置")
    print("2. 可以修改 t_snapshot 来看不同时刻的场分布")
    print("3. 图中可以看到薄膜前后的相位差（虚线标注）")
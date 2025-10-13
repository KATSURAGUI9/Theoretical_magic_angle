import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ============================================================================
# 物理常数
# ============================================================================
c = 299792458.0           # 光速 m/s

# ============================================================================
# 论文参数（Figure 2b）
# ============================================================================
WAVELENGTH = 1030e-9      # 1030 nm
E0 = 2.6e8                  # 1 GV/m
REFRACTIVE_INDEX = 3.6    # Si
MEMBRANE_THICKNESS = 60e-9  # 60 nm

# 角度（论文Figure 2b）
theta_foil = 65        # 度
theta_laser = 45          # 度

# 时间快照（可以调整来看不同相位）
# 对于 λ = 1030 nm，光周期 T = 3.436 fs
# 推荐值：
#   t_snapshot = 0           # 初始相位（当前设置）
#   t_snapshot = 8.59e-16    # T/4 (0.859 fs) - 条纹移动
#   t_snapshot = 1.72e-15    # T/2 (1.718 fs) - 颜色反转
#   t_snapshot = 4.29e-16    # T/8 (0.429 fs) - 轻微移动
t_snapshot = 1.72e-15       # 时间 = 0 (初始)

# ============================================================================
# Fresnel 系数（修正版）
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
# 多次反射求和（改进：返回薄膜内场）
# ============================================================================
def calculate_multiple_reflections(r12, t12, r21, t21, k_film, d, max_order=5):
    """计算薄膜多次反射的累积效应"""
    phi = 2 * k_film * d
    phase_factor = np.exp(1j * phi)

    r_total = r12
    t_total = 0.0
    E_inside_forward = 0.0
    E_inside_backward = 0.0

    for m in range(max_order):
        internal_bounces = (r21**2)**m * phase_factor**(m+1)

        # 反射贡献
        r_total += t12 * r21 * t21 * internal_bounces / phase_factor

        # 透射贡献
        t_total += t12 * t21 * internal_bounces

        # 薄膜内部场（正向和反向波的叠加）
        E_inside_forward += t12 * (r21**2)**m * phase_factor**m
        E_inside_backward += t12 * r21 * (r21**2)**m * phase_factor**m

    return r_total, t_total, E_inside_forward, E_inside_backward


# ============================================================================
# 改进的电磁场计算
# ============================================================================
def calculate_em_field_2d_improved(x, z, t, theta_foil, theta_laser,
                                    wavelength, E0, n, d, max_order=5):
    """
    计算2D空间中的电磁场（改进版）
    
    改进：
        1. 正确处理薄膜内部的驻波场
        2. 准确计算透射波的相位
        3. 使用局部坐标系处理倾斜薄膜
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

    # 入射波波矢（在全局坐标系）
    k_in = k0 * np.array([sin_theta_l, cos_theta_l])  # 2D: [kx, kz]

    # 薄膜法向（在全局坐标系）
    n_foil = np.array([sin_theta_f, cos_theta_f])

    # 入射角（相对于薄膜法向）
    cos_theta_in = np.abs(np.dot(k_in/k0, n_foil))
    theta_in = np.arccos(cos_theta_in)

    # 计算 Fresnel 系数和多次反射
    r12, t12, R12, T12 = fresnel_coefficients(theta_in, n, 'p')
    theta_t = np.arcsin(np.sin(theta_in) / n)
    r21, t21, R21, T21 = fresnel_coefficients(theta_t, 1.0/n, 'p')

    # 薄膜中的波矢大小
    k_film = n * k0

    # 多次反射
    r_total, t_total, E_fwd, E_bwd = calculate_multiple_reflections(
        r12, t12, r21, t21, k_film * np.cos(theta_t), d, max_order
    )

    # 坐标变换：全局 → 薄膜局部坐标
    x_local = x * cos_theta_f - z * sin_theta_f
    z_local = x * sin_theta_f + z * cos_theta_f

    # 判断点的位置
    mask_before = z_local < -d/2      # 薄膜前（入射侧）
    mask_inside = (z_local >= -d/2) & (z_local <= d/2)  # 薄膜内
    mask_after = z_local > d/2        # 薄膜后（透射侧）

    # 初始化电场
    Ex = np.zeros_like(z, dtype=complex)
    Ez = np.zeros_like(z, dtype=complex)

    # ========================================================================
    # 区域1：薄膜前（入射波 + 反射波）
    # ========================================================================
    if np.any(mask_before):
        # 入射波电场振幅（p偏振：在xz平面内）
        E_in_x = E0 * cos_theta_l
        E_in_z = -E0 * sin_theta_l

        # 反射波电场振幅
        E_ref_x = r_total * E0 * cos_theta_l
        E_ref_z = r_total * E0 * sin_theta_l  # z分量符号反转

        # 反射波波矢
        k_ref = k0 * np.array([sin_theta_l, -cos_theta_l])

        # 相位
        phase_in = k_in[0]*x + k_in[1]*z - omega*t
        phase_ref = k_ref[0]*x + k_ref[1]*z - omega*t

        # 总场
        Ex[mask_before] = (E_in_x * np.exp(1j*phase_in[mask_before]) +
                           E_ref_x * np.exp(1j*phase_ref[mask_before]))
        Ez[mask_before] = (E_in_z * np.exp(1j*phase_in[mask_before]) +
                           E_ref_z * np.exp(1j*phase_ref[mask_before]))

    # ========================================================================
    # 区域2：薄膜内（正向波 + 反向波的驻波）
    # ========================================================================
    if np.any(mask_inside):
        # 薄膜内的波矢（折射后）
        theta_eff = theta_t + theta_f
        k_in_film = n * k0 * np.array([np.sin(theta_eff), np.cos(theta_eff)])
        k_ref_film = n * k0 * np.array([np.sin(theta_eff), -np.cos(theta_eff)])

        # 电场振幅（考虑折射率）
        E_fwd_x = E_fwd * E0 * np.cos(theta_eff)
        E_fwd_z = -E_fwd * E0 * np.sin(theta_eff)

        E_bwd_x = E_bwd * E0 * np.cos(theta_eff)
        E_bwd_z = E_bwd * E0 * np.sin(theta_eff)

        # 相位（使用薄膜内的波矢）
        phase_fwd = k_in_film[0]*x + k_in_film[1]*z - omega*t
        phase_bwd = k_ref_film[0]*x + k_ref_film[1]*z - omega*t

        Ex[mask_inside] = (E_fwd_x * np.exp(1j*phase_fwd[mask_inside]) +
                           E_bwd_x * np.exp(1j*phase_bwd[mask_inside]))
        Ez[mask_inside] = (E_fwd_z * np.exp(1j*phase_fwd[mask_inside]) +
                           E_bwd_z * np.exp(1j*phase_bwd[mask_inside]))

    # ========================================================================
    # 区域3：薄膜后（透射波）
    # ========================================================================
    if np.any(mask_after):
        # 透射波方向（延续入射方向，但考虑相位延迟）
        k_trans = k_in

        # 透射波电场振幅
        E_trans_x = t_total * E0 * cos_theta_l
        E_trans_z = -t_total * E0 * sin_theta_l

        # 相位（包含通过薄膜的相位累积）
        phi_membrane = n * k0 * d * np.cos(theta_t)
        phase_trans = (k_trans[0]*x + k_trans[1]*z - omega*t +
                       phi_membrane + np.angle(t_total))

        Ex[mask_after] = E_trans_x * np.exp(1j*phase_trans[mask_after])
        Ez[mask_after] = E_trans_z * np.exp(1j*phase_trans[mask_after])

    # 返回实部（电场的瞬时值）
    return Ex.real, Ez.real


# ============================================================================
# 改进的可视化
# ============================================================================
def plot_field_distribution_improved():
    """绘制电场分布（改进版）"""

    # 创建2D网格
    x_range = np.linspace(-3e-6, 3e-6, 400)  # 更高分辨率
    z_range = np.linspace(-3e-6, 3e-6, 400)
    X, Z = np.meshgrid(x_range, z_range)

    print("计算电场分布（改进版）...")
    # 计算电场
    Ex, Ez = calculate_em_field_2d_improved(
        X, Z, t_snapshot,
        theta_foil, theta_laser,
        WAVELENGTH, E0, REFRACTIVE_INDEX,
        MEMBRANE_THICKNESS, max_order=5
    )

    # 计算电场强度
    E_magnitude = np.sqrt(Ex**2 + Ez**2)
    
    # 【关键修复1】归一化电场
    Ex_norm = Ex / E0
    Ez_norm = Ez / E0
    E_magnitude_norm = E_magnitude / E0

    # 【关键修复2】正确的p偏振场表示（显示总场强）
    #
    # 问题：总场是矢量 E = Ex*x̂ + Ez*ẑ，如何在2D图上显示？
    #
    # 方案对比：
    #   方案1: E_display = Ex_norm
    #          - 优点：简单，保留符号
    #          - 缺点：只显示一个分量，丢失 Ez 信息（对65°角，Ez占91%！）
    #
    #   方案2: E_display = E_magnitude_norm
    #          - 优点：显示完整场强 |E| = √(Ex² + Ez²)
    #          - 缺点：丢失相位信息（正负符号），看不到波的振荡
    #
    #   方案3: E_display = sign(Ex) × E_magnitude_norm  ✓ 推荐
    #          - 优点：包含 Ex 和 Ez 的总场强 + 保留相位（正负符号）
    #          - 缺点：物理意义略模糊（用 Ex 的符号标记总场强）
    #
    #   方案4: E_display = sign(主导分量) × E_magnitude_norm
    #          - 根据角度选择 Ex 或 Ez 作为符号来源

    # ✓ 使用方案3：带符号的总场强
    E_display = np.sign(Ex_norm) * E_magnitude_norm  # 总场强（Ex² + Ez²），用 Ex 的符号

    # ========================================================================
    # 创建图形（2个子图）
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ------------------------------------------------------------------------
    # 子图1：归一化电场（与论文一致）
    # ------------------------------------------------------------------------
    ax = axes[0]

    levels = np.linspace(-1, 1, 51)  # 归一化：-1到+1
    im1 = ax.contourf(Z*1e6, X*1e6, E_display,
                      levels=levels, cmap='RdBu_r', extend='both')

    # 绘制薄膜
    theta_f_rad = np.radians(theta_foil)
    membrane_length = 6e-6  # μm
    half_length = membrane_length / 2
    half_thick = MEMBRANE_THICKNESS / 2
    corners_local = np.array([
        [-half_thick, -half_length],
        [half_thick, -half_length],
        [half_thick, half_length],
        [-half_thick, half_length]
    ])

    # 旋转到全局坐标
    rotation_matrix = np.array([
        [np.cos(theta_f_rad), -np.sin(theta_f_rad)],
        [np.sin(theta_f_rad), np.cos(theta_f_rad)]
    ])
    corners_global = corners_local @ rotation_matrix.T

    # 绘制薄膜（填充矩形）
    membrane_patch = Polygon(corners_global[:, [1, 0]] * 1e6,  # [z, x]
                             facecolor='gray', edgecolor='black',
                             linewidth=2, alpha=0.5, zorder=10)
    ax.add_patch(membrane_patch)

    # 标注激光方向（箭头）
    arrow_start_z = -2.3e-6
    arrow_start_x = -2.3e-6
    theta_l_rad = np.radians(theta_laser)
    arrow_length = 1.2e-6
    arrow_dz = arrow_length * np.cos(theta_l_rad)
    arrow_dx = arrow_length * np.sin(theta_l_rad)
    ax.arrow(arrow_start_z*1e6, arrow_start_x*1e6,
             arrow_dz*1e6, arrow_dx*1e6,
             head_width=0.2, head_length=0.15, fc='red', ec='red',
             linewidth=2.5, zorder=15)
    ax.text(arrow_start_z*1e6-0.4, arrow_start_x*1e6-0.5,
            f'Laser\n{theta_laser}°', fontsize=11, color='red',
            fontweight='bold', zorder=15)

    # 标注薄膜角度
    ax.text(0.5, 2.5, f'Membrane\n{theta_foil}°', fontsize=11,
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 设置坐标轴
    ax.set_xlabel('z (μm)', fontsize=13)
    ax.set_ylabel('x (μm)', fontsize=13)
    ax.set_title(f'Normalized Electric Field (signed |E|/E₀)\n' +
                 f't={t_snapshot*1e15:.0f} as, λ={WAVELENGTH*1e9:.0f} nm, ' +
                 f'Si {MEMBRANE_THICKNESS*1e9:.0f} nm',
                 fontsize=13, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('E / E₀', fontsize=12)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # ------------------------------------------------------------------------
    # 子图2：归一化场强 |E|
    # ------------------------------------------------------------------------
    ax = axes[1]

    levels_mag = np.linspace(0, 1.5, 31)  # 归一化到0-1.5
    im2 = ax.contourf(Z*1e6, X*1e6, E_magnitude_norm,
                      levels=levels_mag, cmap='hot', extend='max')

    # 再次绘制薄膜
    membrane_patch2 = Polygon(corners_global[:, [1, 0]] * 1e6,
                              facecolor='cyan', edgecolor='blue',
                              linewidth=2, alpha=0.4, zorder=10)
    ax.add_patch(membrane_patch2)

    # 绘制等相位线（波峰）
    omega = 2 * np.pi * c / WAVELENGTH
    k0 = omega / c
    k_in = k0 * np.array([np.sin(theta_l_rad), np.cos(theta_l_rad)])

    for i in range(-4, 5):
        # 等相位线：k·r = 2πi
        phase_const = 2 * np.pi * i
        # kx*x + kz*z = const
        # z = (const - kx*x) / kz
        x_line = np.linspace(-3e-6, 3e-6, 100)
        z_line = (phase_const - k_in[0]*x_line) / k_in[1]

        # 只绘制在图内的部分
        mask = (z_line >= -3e-6) & (z_line <= 3e-6)
        if np.any(mask):
            ax.plot(z_line[mask]*1e6, x_line[mask]*1e6,
                   'c--', linewidth=0.8, alpha=0.6, zorder=5)

    # 标注
    ax.set_xlabel('z (μm)', fontsize=13)
    ax.set_ylabel('x (μm)', fontsize=13)
    ax.set_title(f'Normalized Field Magnitude |E|/E₀\n' +
                 f'with Phase Fronts (cyan dashed)',
                 fontsize=13, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('|E| / E₀', fontsize=12)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('field_distribution_PRA_Fig2b_FIXED.png', dpi=300, bbox_inches='tight')
    print(f"已保存图像: field_distribution_PRA_Fig2b_FIXED.png")
    plt.show()

    # 验证能量守恒
    theta_in = np.arccos(np.abs(np.dot(
        np.array([np.sin(np.radians(theta_laser)), np.cos(np.radians(theta_laser))]),
        np.array([np.sin(np.radians(theta_foil)), np.cos(np.radians(theta_foil))])
    )))
    r, t, R, T = fresnel_coefficients(theta_in, REFRACTIVE_INDEX, 'p')

    print(f"\n物理验证：")
    print(f"  入射角（相对薄膜）: {np.degrees(theta_in):.2f}°")
    print(f"  反射系数 r = {r:.6f}")
    print(f"  透射系数 t = {t:.6f}")
    print(f"  能量守恒 R + T = {R + T:.6f} (应该 = 1.000)")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("复现 PRA 论文 Figure 2(b) - 完全修复版")
    print(f"参数:")
    print(f"  波长: {WAVELENGTH*1e9:.0f} nm")
    print(f"  电场: {E0*1e-9:.1f} GV/m")
    print(f"  折射率: {REFRACTIVE_INDEX}")
    print(f"  厚度: {MEMBRANE_THICKNESS*1e9:.0f} nm")
    print(f"  θ_foil: {theta_foil}°")
    print(f"  θ_laser: {theta_laser}°")
    print("="*60)

    plot_field_distribution_improved()

    print("\n✅ 修复要点:")
    print("1. ✓ 归一化显示：-1 到 +1（而不是绝对值）")
    print("2. ✓ 显示带符号的总场强（而不是单一Ex分量）")
    print("3. ✓ 正确计算p偏振的矢量干涉")
    print("4. ✓ 可视化等相位线（波峰位置）")
    print("5. ✓ 验证能量守恒")
    print("\n💡 现在应该能看到类似论文的块状干涉图样！")
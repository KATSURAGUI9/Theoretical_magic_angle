import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============================================================================
# 物理常数
# ============================================================================
c = 299792458.0           # 光速 m/s
e = 1.602176634e-19       # 电子电荷 C
m_e = 9.1093837015e-31    # 电子质量 kg
hbar = 1.054571817e-34    # 约化普朗克常数

# ============================================================================
# 系统参数（可配置）
# ============================================================================
# 电子参数
E_kin = 70e3 * e          # 70 keV
gamma = 1 + E_kin / (m_e * c**2)  # 洛伦兹因子
v0 = c * np.sqrt(1 - 1/gamma**2)  # 电子速度
p0 = gamma * m_e * v0     # 电子动量

# 薄膜参数
REFRACTIVE_INDEX = 3.6    # Si 在近红外的折射率
MEMBRANE_THICKNESS = 60e-9  # 60 nm

# 激光参数
WAVELENGTH = 1030e-9      # 1030 nm
E0_DEFAULT = 1e9          # 默认电场强度 V/m

# 数值计算参数
N_PERIODS = 10            # 积分时长（光周期数）
POINTS_PER_PERIOD = 100   # 每周期采样点数
MAX_REFLECTION_ORDER = 5  # 多次反射展开阶数

print(f"=" * 60)
print(f"电子参数：")
print(f"  动能: {E_kin/e/1e3:.1f} keV")
print(f"  速度: {v0/c:.6f}c")
print(f"  洛伦兹因子: {gamma:.6f}")
print(f"  动量: {p0:.6e} kg·m/s")
print(f"=" * 60)

# ============================================================================
# Fresnel 系数（修正版）
# ============================================================================
def fresnel_coefficients(theta_in, n, polarization='p'):
    """
    计算 Fresnel 反射和透射系数（修正版）

    参数:
        theta_in: 入射角（弧度）
        n: 折射率（n_transmission / n_incident）
        polarization: 'p' 或 's'

    返回:
        r: 反射系数
        t: 透射系数
        R: 反射率
        T: 透射率
    """
    cos_theta_in = np.cos(theta_in)
    sin_theta_in = np.sin(theta_in)

    # Snell 定律
    sin_theta_t = sin_theta_in / n

    # 检查全反射
    if np.abs(sin_theta_t) > 1:
        return -1.0, 0.0, 1.0, 0.0  # 全反射

    cos_theta_t = np.sqrt(1 - sin_theta_t**2)

    if polarization == 'p':
        # p 偏振（TM）
        r = (n * cos_theta_in - cos_theta_t) / (n * cos_theta_in + cos_theta_t)
        t = 2 * cos_theta_in / (n * cos_theta_in + cos_theta_t)
        # 能量守恒：R + T = 1（正确形式）
        R = np.abs(r)**2
        T = np.abs(t)**2 * (n * cos_theta_t) / cos_theta_in
    else:
        # s 偏振（TE）
        r = (cos_theta_in - n * cos_theta_t) / (cos_theta_in + n * cos_theta_t)
        t = 2 * cos_theta_in / (cos_theta_in + n * cos_theta_t)
        R = np.abs(r)**2
        T = np.abs(t)**2 * (n * cos_theta_t) / cos_theta_in

    return r, t, R, T


# 测试能量守恒
theta = np.radians(45)
r_p, t_p, R_p, T_p = fresnel_coefficients(theta, n=3.6, polarization='p')
print(f"\nFresnel 系数测试（45°, Si）：")
print(f"  r_p = {r_p:.6f}")
print(f"  t_p = {t_p:.6f}")
print(f"  R_p = {R_p:.6f}")
print(f"  T_p = {T_p:.6f}")
print(f"  R + T = {R_p + T_p:.6f}  (应该 ≈ 1.000)")

# ============================================================================
# 多次反射求和（5阶展开）
# ============================================================================
def calculate_multiple_reflections(r12, t12, r21, t21, k_film, d, max_order=5):
    """
    计算薄膜多次反射的累积效应

    物理模型：
        - r12: 1->2 界面反射系数（空气->薄膜）
        - t12: 1->2 界面透射系数
        - r21: 2->1 界面反射系数（薄膜->空气）
        - t21: 2->1 界面透射系数
        - k_film: 薄膜中的波矢大小
        - d: 薄膜厚度
        - max_order: 展开阶数

    返回：
        r_total: 总反射系数
        t_total: 总透射系数
    """
    # 薄膜中单次往返的相位
    phi = 2 * k_film * d
    phase_factor = np.exp(1j * phi)

    # 反射系数：几何级数求和
    # r_total = r12 + t12 * r21 * phase_factor * t21 / (1 - r21^2 * phase_factor)
    # 展开为：r12 + t12*t21*r21*e^(iφ) + t12*t21*r21^3*e^(i2φ) + ...

    r_total = r12
    t_total = 0.0

    for m in range(max_order):
        # 第 m 阶内部反射
        # 经历：t12 进入，在薄膜内反射 m 次（每次 r21^2），最后 t21 出射
        internal_bounces = (r21**2)**m * phase_factor**(m+1)

        # 对反射的贡献
        r_total += t12 * r21 * t21 * internal_bounces / phase_factor

        # 对透射的贡献
        t_total += t12 * t21 * internal_bounces

    return r_total, t_total


# ============================================================================
# 电磁场计算（改进版：多次反射 + 正确相位）
# ============================================================================
def em_field_before_membrane(z, t, theta_foil, theta_laser,
                             wavelength, E0, n, d, max_order=5):
    """
    计算薄膜前（z < 0）的电磁场（改进版）

    改进：
        1. 修正了磁场的相位计算
        2. 实现了多次反射的无穷级数求和（展开到 max_order 阶）
        3. 预计算三角函数值以提高性能

    参数:
        z: 位置（沿电子轨迹）
        t: 时间
        theta_foil: 薄膜角度（度）
        theta_laser: 激光角度（度）
        wavelength: 波长（m）
        E0: 入射场强（V/m）
        n: 折射率
        d: 薄膜厚度（m）
        max_order: 多次反射展开阶数

    返回:
        E: 电场 [Ex, Ey, Ez]
        B: 磁场 [Bx, By, Bz]
    """
    # 转换为弧度（预计算）
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)

    cos_theta_f = np.cos(theta_f)
    sin_theta_f = np.sin(theta_f)
    cos_theta_l = np.cos(theta_l)
    sin_theta_l = np.sin(theta_l)

    omega = 2 * np.pi * c / wavelength
    k0 = omega / c

    # 入射波波矢（在 xz 平面）
    k_in = k0 * np.array([sin_theta_l, 0, cos_theta_l])

    # 薄膜法向
    n_foil = np.array([sin_theta_f, 0, cos_theta_f])

    # 入射角（相对于薄膜法向）
    theta_in = np.arccos(np.abs(np.dot(k_in/k0, n_foil)))

    # 计算 Fresnel 系数
    r12, t12, R12, T12 = fresnel_coefficients(theta_in, n, 'p')

    # 薄膜内部的折射角
    theta_t = np.arcsin(np.sin(theta_in) / n)

    # 内部界面（薄膜->空气）
    r21, t21, R21, T21 = fresnel_coefficients(theta_t, 1.0/n, 'p')

    # 薄膜中的波矢大小
    k_film = n * k0 * np.cos(theta_t)

    # 计算多次反射的总系数
    r_total, t_total = calculate_multiple_reflections(
        r12, t12, r21, t21, k_film, d, max_order
    )

    # ========================================================================
    # 构造电磁场（振幅和相位分离）
    # ========================================================================

    # 入射波电场振幅（p偏振：在入射平面内，垂直于 k）
    E_in_amp = E0 * np.array([cos_theta_l, 0, -sin_theta_l])

    # 反射波电场振幅
    E_ref_amp = r_total * E0 * np.array([cos_theta_l, 0, sin_theta_l])

    # 磁场振幅：B = (k × E) / omega
    B_in_amp = np.cross(k_in, E_in_amp) / omega

    # 反射波波矢
    k_ref = k0 * np.array([sin_theta_l, 0, -cos_theta_l])
    B_ref_amp = np.cross(k_ref, E_ref_amp) / omega

    # 计算相位
    phase_in = np.dot(k_in, [0, 0, z]) - omega * t
    phase_ref = np.dot(k_ref, [0, 0, z]) - omega * t

    # 实部：真实场
    if np.iscomplex(r_total):
        # 复数反射系数：需要考虑相位
        E_ref = (E_ref_amp * np.exp(1j * phase_ref)).real
        B_ref = (B_ref_amp * np.exp(1j * phase_ref)).real
    else:
        E_ref = E_ref_amp * np.cos(phase_ref)
        B_ref = B_ref_amp * np.cos(phase_ref)

    E_in = E_in_amp * np.cos(phase_in)
    B_in = B_in_amp * np.cos(phase_in)

    # 总场
    E = E_in + E_ref
    B = B_in + B_ref

    return E, B


# 测试改进后的电磁场
E, B = em_field_before_membrane(
    z=-1e-6,  # -1 μm
    t=0,
    theta_foil=35,
    theta_laser=155,
    wavelength=WAVELENGTH,
    E0=E0_DEFAULT,
    n=REFRACTIVE_INDEX,
    d=MEMBRANE_THICKNESS,
    max_order=MAX_REFLECTION_ORDER
)

print(f"\n电磁场测试（含5阶多次反射）：")
print(f"  E = [{E[0]:.3e}, {E[1]:.3e}, {E[2]:.3e}] V/m")
print(f"  B = [{B[0]:.3e}, {B[1]:.3e}, {B[2]:.3e}] T")
print(f"  |E| = {np.linalg.norm(E):.3e} V/m")
print(f"  |B| = {np.linalg.norm(B):.3e} T")
print(f"  |E|/(c|B|) = {np.linalg.norm(E)/(c*np.linalg.norm(B)):.3f} (理想 = 1)")

# ============================================================================
# 电子运动方程（改进版：使用 RK45 求解器）
# ============================================================================
def equations_of_motion(t, state, theta_foil, theta_laser,
                       wavelength, E0, n, d, max_order):
    """
    电子在电磁场中的运动方程

    state = [z, vx, vy, vz]
    d(state)/dt = [vz, ax, ay, az]

    考虑相对论效应和速度更新
    """
    z, vx, vy, vz = state

    # 获取电磁场
    E, B = em_field_before_membrane(
        z, t, theta_foil, theta_laser,
        wavelength, E0, n, d, max_order
    )

    # 当前速度
    v_vec = np.array([vx, vy, vz])

    # 相对论 gamma 因子
    v_mag = np.linalg.norm(v_vec)
    gamma_instant = 1.0 / np.sqrt(1 - (v_mag/c)**2)

    # 洛伦兹力
    F = -e * (E + np.cross(v_vec, B))

    # 加速度
    a = F / (gamma_instant * m_e)

    return [vz, a[0], a[1], a[2]]


def calculate_alpha_sample_rk45(theta_foil, theta_laser,
                                wavelength=WAVELENGTH, E0=E0_DEFAULT,
                                n=REFRACTIVE_INDEX, d=MEMBRANE_THICKNESS,
                                max_order=MAX_REFLECTION_ORDER):
    """
    使用 RK45 求解器计算 α_sample（改进版）

    改进：
        1. 使用 scipy 的 RK45 求解器（4-5阶龙格-库塔）
        2. 考虑电子速度的变化
        3. 更高的时间分辨率

    返回:
        alpha_sample: 偏转角（弧度）
        alpha_x, alpha_y: 各分量
        success: 是否计算成功
    """
    omega = 2 * np.pi * c / wavelength
    T = 2 * np.pi / omega  # 周期

    # 初始条件：[z, vx, vy, vz]
    z0 = -N_PERIODS * T * v0  # 从远处开始
    y0 = [z0, 0.0, 0.0, v0]

    # 时间范围
    t_span = (-N_PERIODS * T, 0)

    # 评估点（用于输出）
    t_eval = np.linspace(-N_PERIODS * T, 0, N_PERIODS * POINTS_PER_PERIOD)

    try:
        # 使用 RK45 求解
        sol = solve_ivp(
            equations_of_motion,
            t_span,
            y0,
            args=(theta_foil, theta_laser, wavelength, E0, n, d, max_order),
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            logging.warning(f"RK45 求解失败：{sol.message}")
            return np.nan, np.nan, np.nan, False

        # 提取最终速度
        vx_final = sol.y[1, -1]
        vy_final = sol.y[2, -1]
        vz_final = sol.y[3, -1]

        # 计算最终动量
        v_final = np.array([vx_final, vy_final, vz_final])
        v_mag_final = np.linalg.norm(v_final)
        gamma_final = 1.0 / np.sqrt(1 - (v_mag_final/c)**2)
        p_final = gamma_final * m_e * v_final

        # 动量变化
        p_initial = np.array([0, 0, p0])
        dp = p_final - p_initial

        # 偏转角
        alpha_x = dp[0] / p0
        alpha_y = dp[1] / p0
        alpha_sample = np.sqrt(alpha_x**2 + alpha_y**2)

        return alpha_sample, alpha_x, alpha_y, True

    except Exception as e:
        logging.warning(f"计算失败: {e}")
        return np.nan, np.nan, np.nan, False


# 测试 RK45 求解器
print(f"\n" + "="*60)
print(f"测试 RK45 求解器（这可能需要几秒钟）...")
print(f"="*60)

alpha, alpha_x, alpha_y, success = calculate_alpha_sample_rk45(
    theta_foil=35,
    theta_laser=155
)

if success:
    print(f"\n单点测试（RK45 + 5阶多次反射）：")
    print(f"  θ_foil = 35°, θ_laser = 155°")
    print(f"  α_sample = {alpha*1e3:.6f} mrad")
    print(f"  α_x = {alpha_x*1e3:.6f} mrad")
    print(f"  α_y = {alpha_y*1e3:.6f} mrad")
else:
    print(f"\n单点测试失败！")

# ============================================================================
# 参数空间扫描（改进版：更好的错误处理）
# ============================================================================
def scan_parameter_space(theta_foil_range, theta_laser_range,
                         wavelength=WAVELENGTH, E0=E0_DEFAULT,
                         n=REFRACTIVE_INDEX, d=MEMBRANE_THICKNESS,
                         max_order=MAX_REFLECTION_ORDER):
    """
    扫描参数空间，寻找魔角（改进版）

    改进：
        1. 更精确的错误处理
        2. 记录失败的计算

    返回:
        alpha_map: 2D 数组
    """
    Nf = len(theta_foil_range)
    Nl = len(theta_laser_range)

    alpha_map = np.zeros((Nf, Nl))
    failed_count = 0

    print(f"\n扫描参数空间...")
    print(f"  θ_foil: {Nf} 点")
    print(f"  θ_laser: {Nl} 点")
    print(f"  总计: {Nf * Nl} 点")
    print(f"  多次反射阶数: {max_order}")

    for i, theta_f in enumerate(tqdm(theta_foil_range, desc="扫描进度")):
        for j, theta_l in enumerate(theta_laser_range):

            alpha, _, _, success = calculate_alpha_sample_rk45(
                theta_f, theta_l,
                wavelength, E0, n, d, max_order
            )

            if success:
                alpha_map[i, j] = alpha
            else:
                alpha_map[i, j] = np.nan
                failed_count += 1

    if failed_count > 0:
        logging.warning(f"有 {failed_count}/{Nf*Nl} 个点计算失败")

    return alpha_map


# ============================================================================
# 可视化和候选点查找（与原版相同）
# ============================================================================
def visualize_and_find_candidates(alpha_map, theta_foil_array, theta_laser_array):
    """
    可视化 α_sample map，并找到候选魔角
    """
    # 转换为 mrad
    alpha_map_mrad = alpha_map * 1e3

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 2D colormap
    ax = axes[0]
    im = ax.contourf(theta_laser_array, theta_foil_array, alpha_map_mrad,
                     levels=20, cmap='viridis')
    ax.set_xlabel('θ_laser (degrees)', fontsize=12)
    ax.set_ylabel('θ_foil (degrees)', fontsize=12)
    ax.set_title('α_sample (mrad) - Improved Model', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('α_sample (mrad)', fontsize=11)

    # 标出最小值点
    min_idx = np.unravel_index(np.nanargmin(alpha_map), alpha_map.shape)
    theta_f_min = theta_foil_array[min_idx[0]]
    theta_l_min = theta_laser_array[min_idx[1]]
    alpha_min = alpha_map[min_idx]

    ax.plot(theta_l_min, theta_f_min, 'r*', markersize=20,
            label=f'Min: ({theta_f_min:.1f}°, {theta_l_min:.1f}°)')
    ax.legend(fontsize=10)

    # (b) 沿某个切面的 1D plot
    ax = axes[1]

    # 固定 θ_laser，扫描 θ_foil
    j_fixed = np.argmin(np.abs(theta_laser_array - theta_l_min))

    ax.plot(theta_foil_array, alpha_map_mrad[:, j_fixed], 'o-', lw=2)
    ax.axhline(0, color='r', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('θ_foil (degrees)', fontsize=12)
    ax.set_ylabel('α_sample (mrad)', fontsize=12)
    ax.set_title(f'1D cut at θ_laser = {theta_laser_array[j_fixed]:.1f}°', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alpha_sample_map_improved.png', dpi=300)
    print(f"\n已保存图像: alpha_sample_map_improved.png")
    plt.show()

    # 找到所有候选点（α < 阈值）
    threshold = np.nanpercentile(alpha_map, 10)  # 最小的 10%

    candidates = []
    for i in range(len(theta_foil_array)):
        for j in range(len(theta_laser_array)):
            if not np.isnan(alpha_map[i, j]) and alpha_map[i, j] < threshold:
                candidates.append({
                    'theta_foil': theta_foil_array[i],
                    'theta_laser': theta_laser_array[j],
                    'alpha_sample': alpha_map[i, j]
                })

    # 按 α_sample 排序
    candidates = sorted(candidates, key=lambda x: x['alpha_sample'])

    print(f"\n找到 {len(candidates)} 个候选魔角：")
    print(f"\nTop 10:")
    print(f"{'Rank':<6} {'θ_foil':<10} {'θ_laser':<10} {'α_sample (mrad)':<15}")
    print("-" * 45)

    for rank, cand in enumerate(candidates[:10], 1):
        print(f"{rank:<6} {cand['theta_foil']:>8.1f}°  {cand['theta_laser']:>8.1f}°  "
              f"{cand['alpha_sample']*1e3:>13.6f}")

    return candidates


def fine_scan_around_candidate(theta_f_center, theta_l_center,
                               d_theta=5, n_points=21,
                               **kwargs):
    """
    在候选点附近精细扫描
    """
    theta_foil_fine = np.linspace(
        theta_f_center - d_theta,
        theta_f_center + d_theta,
        n_points
    )

    theta_laser_fine = np.linspace(
        theta_l_center - d_theta,
        theta_l_center + d_theta,
        n_points
    )

    print(f"\n" + "="*60)
    print(f"精细扫描：")
    print(f"  中心：({theta_f_center:.1f}°, {theta_l_center:.1f}°)")
    print(f"  范围：±{d_theta}°")
    print(f"  步长：{2*d_theta/(n_points-1):.2f}°")
    print(f"="*60)

    # 扫描
    alpha_map_fine = scan_parameter_space(
        theta_foil_fine,
        theta_laser_fine,
        **kwargs
    )

    # 找最小值
    min_idx = np.unravel_index(np.nanargmin(alpha_map_fine), alpha_map_fine.shape)
    theta_f_opt = theta_foil_fine[min_idx[0]]
    theta_l_opt = theta_laser_fine[min_idx[1]]
    alpha_opt = alpha_map_fine[min_idx]

    print(f"\n✨ 精细扫描结果：")
    print(f"  最优角度：({theta_f_opt:.2f}°, {theta_l_opt:.2f}°)")
    print(f"  α_sample = {alpha_opt*1e3:.6f} mrad")

    # 绘图
    plt.figure(figsize=(8, 6))
    im = plt.contourf(theta_laser_fine, theta_foil_fine, alpha_map_fine*1e3,
                      levels=20, cmap='RdYlGn_r')
    plt.plot(theta_l_opt, theta_f_opt, 'r*', markersize=20)
    plt.xlabel('θ_laser (degrees)', fontsize=12)
    plt.ylabel('θ_foil (degrees)', fontsize=12)
    plt.title('Fine scan of α_sample (mrad) - Improved', fontsize=14)
    plt.colorbar(im, label='α_sample (mrad)')
    plt.tight_layout()
    plt.savefig('alpha_sample_fine_scan_improved.png', dpi=300)
    print(f"已保存图像: alpha_sample_fine_scan_improved.png")
    plt.show()

    return theta_f_opt, theta_l_opt, alpha_opt


# ============================================================================
# 主程序：参数扫描
# ============================================================================
if __name__ == "__main__":
    print(f"\n" + "="*60)
    print(f"开始粗扫描...")
    print(f"="*60)

    # 定义搜索范围（可以先用较粗的网格测试）
    theta_foil_array = np.linspace(-60, 60, 31)   # 4° 步长（减少计算量）
    theta_laser_array = np.linspace(0, 180, 46)   # 4° 步长

    # 扫描
    alpha_map = scan_parameter_space(
        theta_foil_array,
        theta_laser_array,
        wavelength=WAVELENGTH,
        E0=E0_DEFAULT,
        n=REFRACTIVE_INDEX,
        d=MEMBRANE_THICKNESS,
        max_order=MAX_REFLECTION_ORDER
    )

    print(f"\n扫描完成！")
    print(f"  α_sample 范围: [{np.nanmin(alpha_map)*1e3:.6f}, {np.nanmax(alpha_map)*1e3:.6f}] mrad")

    # 可视化并找候选点
    candidates = visualize_and_find_candidates(
        alpha_map,
        theta_foil_array,
        theta_laser_array
    )

    # 对第一个候选点进行精细扫描
    if len(candidates) > 0:
        best_candidate = candidates[0]

        theta_f_opt, theta_l_opt, alpha_opt = fine_scan_around_candidate(
            theta_f_center=best_candidate['theta_foil'],
            theta_l_center=best_candidate['theta_laser'],
            d_theta=3,        # ±3°
            n_points=21,      # 0.3° 步长
            wavelength=WAVELENGTH,
            E0=E0_DEFAULT,
            n=REFRACTIVE_INDEX,
            d=MEMBRANE_THICKNESS,
            max_order=MAX_REFLECTION_ORDER
        )

        print(f"\n" + "="*60)
        print(f"最终结果：")
        print(f"  最佳魔角配置：")
        print(f"    θ_foil  = {theta_f_opt:.3f}°")
        print(f"    θ_laser = {theta_l_opt:.3f}°")
        print(f"  最小偏转角：")
        print(f"    α_sample = {alpha_opt*1e3:.6f} mrad")
        print(f"              = {alpha_opt*1e6:.3f} μrad")
        print(f"="*60)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap
import time

# 物理常数
c = 299792458.0  # 光速 m/s
e = 1.602e-19    # 电子电荷 C
m_e = 9.109e-31  # 电子质量 kg

# 默认参数
DEFAULT_WAVELENGTH = 1030e-9  # m
DEFAULT_E0 = 1e9              # V/m
DEFAULT_n = 3.6               # 折射率
DEFAULT_d = 60e-9             # 薄膜厚度 m

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

def calc_field_at_point(x, z, t, theta_foil, theta_laser, wavelength, E0, n, d):
    """
    计算空间点(x,z)在时刻t的电场

    参数:
    - x, z: 位置 (m)
    - t: 时间 (s)
    - theta_foil: 薄膜法向角度 (度)
    - theta_laser: 激光入射角度 (度)
    - wavelength: 激光波长 (m)
    - E0: 电场强度 (V/m)
    - n: 折射率
    - d: 薄膜厚度 (m)

    返回: Ex, Ez (V/m)
    """
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)

    omega = 2 * np.pi * c / wavelength
    k0 = omega / c

    # 入射波矢
    kx_in = k0 * np.sin(theta_l)
    kz_in = k0 * np.cos(theta_l)
    k_in = np.array([kx_in, kz_in])

    # 薄膜法向
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)
    n_foil = np.array([nx, nz])

    # 计算入射角和Fresnel系数
    cos_theta_in = abs(np.dot(k_in, n_foil)) / k0
    theta_in = np.arccos(np.clip(cos_theta_in, 0, 1))

    r12, t12 = fresnel_p(theta_in, n)

    sin_t = np.sin(theta_in) / n
    cos_t = np.sqrt(1 - sin_t**2)
    theta_t = np.arcsin(sin_t)

    r21, t21 = fresnel_p(theta_t, 1/n)

    # 多次反射
    phi = 2 * n * k0 * d * cos_t
    denom = 1 - r21**2 * np.exp(1j * phi)
    r_eff = r12 + t12 * t21 * r21 * np.exp(1j * phi) / denom
    t_eff = t12 * t21 * np.exp(1j * phi/2) / denom

    # 点到薄膜的距离
    dist = x * nx + z * nz

    # 判断区域并计算电场
    if dist < -1e-10:  # 左侧：入射+反射
        phase_in = kx_in * x + kz_in * z - omega * t
        E_in = E0 * np.exp(1j * phase_in)

        # 反射波
        k_dot_n = np.dot(k_in, n_foil)
        k_ref = k_in - 2 * k_dot_n * n_foil
        kx_ref = k_ref[0]
        kz_ref = k_ref[1]

        phase_ref = kx_ref * x + kz_ref * z - omega * t
        E_ref = r_eff * E0 * np.exp(1j * phase_ref)

        E_complex = E_in + E_ref

    elif dist > d + 1e-10:  # 右侧：透射
        phase_trans = kx_in * x + kz_in * z - omega * t
        E_complex = t_eff * E0 * np.exp(1j * phase_trans)

    else:  # 薄膜内
        k_membrane = n * k0
        phase_mem = k_membrane * (kx_in/k0 * x + kz_in/k0 * z) - omega * t
        E_complex = t12 * E0 * np.exp(1j * phase_mem)

    # p偏振：电场在入射平面内
    E_magnitude = np.real(E_complex)
    Ex = E_magnitude * np.cos(theta_l)
    Ez = -E_magnitude * np.sin(theta_l)

    return Ex, Ez

def electron_equation(state, t, theta_foil, theta_laser, wavelength, E0, n, d):
    """
    电子运动方程
    state = [x, z, vx, vz]
    """
    x, z, vx, vz = state

    # 计算当前位置的电场
    Ex, Ez = calc_field_at_point(x, z, t, theta_foil, theta_laser, wavelength, E0, n, d)

    # 加速度
    ax = (e / m_e) * Ex
    az = (e / m_e) * Ez

    return [vx, vz, ax, az]

def calculate_alpha_sample(theta_foil, theta_laser, wavelength=DEFAULT_WAVELENGTH,
                          foil_thickness=DEFAULT_d, field_amplitude=DEFAULT_E0, refractive_index=DEFAULT_n,
                          electron_energy_eV=1000, verbose=False):
    """
    计算电子穿过薄膜中心时的瞬时偏转角 α_sample

    电子从z轴负方向沿+z方向入射（垂直入射）

    返回:
    - alpha_sample: 瞬时偏转角 (度)
    - success: 是否成功穿过薄膜
    """
    # 电子初始速度（沿+z方向）
    E_k = electron_energy_eV * e  # e 是全局常量
    v0 = np.sqrt(2 * E_k / m_e)

    # 使用简短的局部变量名
    d = foil_thickness
    E0 = field_amplitude
    n = refractive_index

    # 初始位置：在薄膜左侧远处，z轴上
    # 需要计算薄膜的位置
    theta_f = np.radians(theta_foil)
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)

    # 薄膜中心在原点，法向为 (nx, nz)
    # 电子从 z 轴负方向入射，初始位置设在 z = -2μm 处
    x0 = 0.0
    z0 = -2e-6

    # 初始速度：沿+z方向
    vx0 = 0.0
    vz0 = v0

    initial_state = [x0, z0, vx0, vz0]

    # 模拟时间要足够长，让电子穿过薄膜
    # 估算到达薄膜所需时间
    initial_dist_to_foil = abs(x0 * nx + z0 * nz)
    time_estimate = initial_dist_to_foil / v0 * 2.0  # 留2倍余量
    t_max = max(2e-13, time_estimate)  # 至少200 fs
    n_steps = 15000
    t_array = np.linspace(0, t_max, n_steps)

    try:
        # 求解轨迹
        trajectory = odeint(electron_equation, initial_state, t_array,
                          args=(theta_foil, theta_laser, wavelength, E0, n, d))

        x_pos = trajectory[:, 0]
        z_pos = trajectory[:, 1]
        vx = trajectory[:, 2]
        vz = trajectory[:, 3]

        # 计算到薄膜的距离
        distances = x_pos * nx + z_pos * nz

        # 找到穿过薄膜中心的点（距离 ≈ d/2）
        membrane_center_dist = d / 2
        idx_center = np.argmin(np.abs(distances - membrane_center_dist))

        # 检查是否真的穿过了薄膜
        if distances[idx_center] < 0 or distances[idx_center] > d:
            if verbose:
                print(f"  电子未穿过薄膜: dist={distances[idx_center]*1e9:.2f} nm")
            return None, False

        # 计算瞬时速度方向
        vx_center = vx[idx_center]
        vz_center = vz[idx_center]

        # 瞬时偏转角：相对于初始方向(+z方向)的偏转
        # tan(alpha) = vx / vz
        alpha_sample_rad = np.arctan2(vx_center, vz_center)
        alpha_sample_deg = np.degrees(alpha_sample_rad)

        if verbose:
            print(f"  薄膜中心: x={x_pos[idx_center]*1e9:.2f} nm, z={z_pos[idx_center]*1e9:.2f} nm")
            print(f"  速度: vx={vx_center:.3e}, vz={vz_center:.3e}")
            print(f"  α_sample = {alpha_sample_deg:.3f}°")

        return alpha_sample_deg, True

    except Exception as err:
        if verbose:
            print(f"  计算失败: {err}")
        return None, False

def scan_parameter_space(theta_foil_range, theta_laser_range,
                        wavelength=DEFAULT_WAVELENGTH, foil_thickness=DEFAULT_d,
                        field_amplitude=DEFAULT_E0, refractive_index=DEFAULT_n, electron_energy_eV=1000):
    """
    扫描参数空间，寻找 α_sample ≈ 0 的条件

    参数:
    - theta_foil_range: 薄膜角度范围 (度)
    - theta_laser_range: 激光角度范围 (度)
    """
    d = foil_thickness
    E0 = field_amplitude
    n = refractive_index

    print("="*70)
    print("寻找Magic Angle条件：α_sample = 0")
    print("="*70)
    print(f"固定参数:")
    print(f"  波长: λ = {wavelength*1e9:.1f} nm")
    print(f"  薄膜厚度: d = {d*1e9:.1f} nm")
    print(f"  折射率: n = {n}")
    print(f"  电子能量: E = {electron_energy_eV} eV")
    print(f"  电场强度: E0 = {E0:.2e} V/m")
    print(f"\n扫描参数:")
    print(f"  θ_foil: {theta_foil_range[0]}° - {theta_foil_range[-1]}° ({len(theta_foil_range)} 点)")
    print(f"  θ_laser: {theta_laser_range[0]}° - {theta_laser_range[-1]}° ({len(theta_laser_range)} 点)")
    print("="*70)

    # 结果存储
    alpha_map = np.full((len(theta_foil_range), len(theta_laser_range)), np.nan)

    start_time = time.time()
    total_points = len(theta_foil_range) * len(theta_laser_range)
    calculated = 0

    print("\n开始扫描...")

    for i, theta_f in enumerate(theta_foil_range):
        for j, theta_l in enumerate(theta_laser_range):
            alpha, success = calculate_alpha_sample(
                theta_f, theta_l, wavelength, d, E0, n, electron_energy_eV, verbose=False
            )

            if success:
                alpha_map[i, j] = alpha
                calculated += 1

            # 进度显示
            if (i * len(theta_laser_range) + j + 1) % 50 == 0:
                elapsed = time.time() - start_time
                progress = (i * len(theta_laser_range) + j + 1) / total_points
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                print(f"  进度: {progress*100:.1f}% | "
                      f"成功: {calculated}/{i * len(theta_laser_range) + j + 1} | "
                      f"耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n扫描完成！总耗时: {elapsed:.1f}s")
    print(f"成功计算: {calculated}/{total_points} ({100*calculated/total_points:.1f}%)")

    return alpha_map

def find_magic_angles(alpha_map, theta_foil_range, theta_laser_range, threshold=0.5):
    """
    从扫描结果中找到 |α_sample| < threshold 的条件

    返回:
    - magic_conditions: [(theta_foil, theta_laser, alpha_sample), ...]
    """
    magic_conditions = []

    for i, theta_f in enumerate(theta_foil_range):
        for j, theta_l in enumerate(theta_laser_range):
            alpha = alpha_map[i, j]
            if not np.isnan(alpha) and abs(alpha) < threshold:
                magic_conditions.append((theta_f, theta_l, alpha))

    # 按 |alpha| 排序
    magic_conditions.sort(key=lambda x: abs(x[2]))

    return magic_conditions

def plot_results(alpha_map, theta_foil_range, theta_laser_range,
                wavelength=DEFAULT_WAVELENGTH, foil_thickness=DEFAULT_d, magic_conditions=None):
    """
    可视化扫描结果
    """
    d = foil_thickness

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左图：α_sample 热力图
    ax = axes[0]

    # 创建网格
    TF, TL = np.meshgrid(theta_foil_range, theta_laser_range, indexing='ij')

    # 绘制热力图
    im = ax.pcolormesh(TL, TF, alpha_map,
                       cmap='RdBu_r', shading='auto',
                       vmin=-5, vmax=5)

    # 标记 α_sample ≈ 0 的点
    if magic_conditions:
        for theta_f, theta_l, alpha in magic_conditions[:10]:  # 只标记前10个
            ax.plot(theta_l, theta_f, 'g*', markersize=15,
                   markeredgecolor='black', markeredgewidth=1.5)

    # 零线
    contours = ax.contour(TL, TF, alpha_map, levels=[0],
                         colors='lime', linewidths=3, linestyles='-')
    ax.clabel(contours, inline=True, fontsize=10)

    ax.set_xlabel('Laser Angle θ_laser (degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Foil Angle θ_foil (degrees)', fontsize=13, fontweight='bold')
    ax.set_title(f'Instantaneous Deflection α_sample\n(λ={wavelength*1e9:.0f}nm, d={d*1e9:.0f}nm)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax, label='α_sample (degrees)')
    cbar.set_label('α_sample (degrees)', fontsize=12, fontweight='bold')

    # 右图：|α_sample| 热力图（对数尺度）
    ax = axes[1]

    alpha_abs = np.abs(alpha_map)
    im2 = ax.pcolormesh(TL, TF, alpha_abs,
                        cmap='viridis_r', shading='auto',
                        norm=plt.matplotlib.colors.LogNorm(vmin=0.01, vmax=10))

    # 标记 magic angle 点
    if magic_conditions:
        for theta_f, theta_l, alpha in magic_conditions[:10]:
            ax.plot(theta_l, theta_f, 'r*', markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5)
            ax.text(theta_l, theta_f + 1.5, f'{abs(alpha):.2f}°',
                   fontsize=9, ha='center', color='white',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    ax.set_xlabel('Laser Angle θ_laser (degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Foil Angle θ_foil (degrees)', fontsize=13, fontweight='bold')
    ax.set_title('|α_sample| - Searching for Magic Angles',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(im2, ax=ax, label='|α_sample| (degrees, log scale)')
    cbar2.set_label('|α_sample| (degrees)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('magic_angle_search.png', dpi=300, bbox_inches='tight')
    print("\n结果已保存到: magic_angle_search.png")

    return fig

def main():
    """主程序"""
    # 定义扫描范围
    theta_foil_range = np.linspace(30, 60, 31)   # 薄膜角度 30-60度
    theta_laser_range = np.linspace(60, 85, 26)  # 激光角度 60-85度

    # 固定参数
    wavelength = DEFAULT_WAVELENGTH
    foil_thickness = DEFAULT_d
    electron_energy_eV = 1000

    # 扫描参数空间
    alpha_map = scan_parameter_space(
        theta_foil_range, theta_laser_range,
        wavelength=wavelength, foil_thickness=foil_thickness, electron_energy_eV=electron_energy_eV
    )

    # 寻找 magic angles
    print("\n" + "="*70)
    print("寻找Magic Angle条件 (|α_sample| < 0.5°)...")
    print("="*70)

    magic_conditions = find_magic_angles(alpha_map, theta_foil_range, theta_laser_range,
                                        threshold=0.5)

    if magic_conditions:
        print(f"\n找到 {len(magic_conditions)} 个Magic Angle条件!")
        print("\n前10个最佳条件:")
        print(f"{'Rank':<6} {'θ_foil':<10} {'θ_laser':<10} {'α_sample':<12}")
        print("-" * 40)
        for i, (theta_f, theta_l, alpha) in enumerate(magic_conditions[:10]):
            print(f"{i+1:<6} {theta_f:<10.2f} {theta_l:<10.2f} {alpha:<12.4f}°")

        # 最佳条件
        best = magic_conditions[0]
        print("\n" + "="*70)
        print("最佳Magic Angle条件:")
        print("="*70)
        print(f"  薄膜角度: θ_foil = {best[0]:.2f}°")
        print(f"  激光角度: θ_laser = {best[1]:.2f}°")
        print(f"  瞬时偏转角: α_sample = {best[2]:.4f}°")
        print(f"\n  固定参数: λ={wavelength*1e9:.0f}nm, d={foil_thickness*1e9:.0f}nm, n={DEFAULT_n}")
        print("="*70)
    else:
        print("\n未找到 |α_sample| < 0.5° 的条件。")
        print("可能需要:")
        print("  1. 扩大扫描范围")
        print("  2. 调整其他参数（波长、厚度等）")
        print("  3. 增加扫描精度")

    # 可视化
    plot_results(alpha_map, theta_foil_range, theta_laser_range,
                wavelength, foil_thickness, magic_conditions)

    return alpha_map, magic_conditions

if __name__ == "__main__":
    alpha_map, magic_conditions = main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

def simulate_electron_trajectory(theta_foil, theta_laser,
                                wavelength=DEFAULT_WAVELENGTH,
                                foil_thickness=DEFAULT_d,
                                field_amplitude=DEFAULT_E0,
                                refractive_index=DEFAULT_n,
                                electron_energy_eV=1000,
                                initial_z=-2e-6,
                                t_max=None,
                                n_steps=20000):
    """
    模拟电子轨迹

    参数:
    - theta_foil: 薄膜角度 (度)
    - theta_laser: 激光角度 (度)
    - wavelength: 激光波长 (m)
    - foil_thickness: 薄膜厚度 (m)
    - field_amplitude: 电场强度 (V/m)
    - refractive_index: 折射率
    - electron_energy_eV: 电子能量 (eV)
    - initial_z: 初始z位置 (m)
    - t_max: 最大模拟时间 (s)，如果为None则自动计算
    - n_steps: 时间步数

    返回:
    - t_array: 时间数组
    - trajectory: [x, z, vx, vz] 轨迹数组
    - info: 包含各种信息的字典
    """
    d = foil_thickness
    E0 = field_amplitude
    n = refractive_index

    # 电子初始速度
    E_k = electron_energy_eV * e
    v0 = np.sqrt(2 * E_k / m_e)

    # 薄膜法向
    theta_f = np.radians(theta_foil)
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)

    # 初始条件：从z轴负方向沿+z方向入射
    x0 = 0.0
    z0 = initial_z
    vx0 = 0.0
    vz0 = v0

    initial_state = [x0, z0, vx0, vz0]

    # 自动计算模拟时间
    if t_max is None:
        initial_dist_to_foil = abs(x0 * nx + z0 * nz)
        time_estimate = initial_dist_to_foil / v0 * 2.5
        t_max = max(2.5e-13, time_estimate)

    t_array = np.linspace(0, t_max, n_steps)

    # 求解轨迹
    print(f"模拟参数:")
    print(f"  薄膜角度: θ_foil = {theta_foil}°")
    print(f"  激光角度: θ_laser = {theta_laser}°")
    print(f"  电子能量: E = {electron_energy_eV} eV")
    print(f"  电子速度: v = {v0:.3e} m/s ({v0/c:.4f}c)")
    print(f"  初始位置: ({x0*1e6:.2f}, {z0*1e6:.2f}) μm")
    print(f"  模拟时间: {t_max*1e15:.1f} fs")
    print(f"\n求解轨迹...")

    trajectory = odeint(electron_equation, initial_state, t_array,
                       args=(theta_foil, theta_laser, wavelength, E0, n, d))

    x_pos = trajectory[:, 0]
    z_pos = trajectory[:, 1]
    vx = trajectory[:, 2]
    vz = trajectory[:, 3]

    # 计算到薄膜的距离
    distances = x_pos * nx + z_pos * nz

    # 找到薄膜中心的点
    membrane_center_dist = d / 2
    idx_center = np.argmin(np.abs(distances - membrane_center_dist))

    # 计算瞬时偏转角
    vx_center = vx[idx_center]
    vz_center = vz[idx_center]
    alpha_sample_rad = np.arctan2(vx_center, vz_center)
    alpha_sample_deg = np.degrees(alpha_sample_rad)

    # 收集信息
    info = {
        'theta_foil': theta_foil,
        'theta_laser': theta_laser,
        'wavelength': wavelength,
        'foil_thickness': d,
        'field_amplitude': E0,
        'refractive_index': n,
        'electron_energy_eV': electron_energy_eV,
        'v0': v0,
        'nx': nx,
        'nz': nz,
        'alpha_sample': alpha_sample_deg,
        'center_position': (x_pos[idx_center], z_pos[idx_center]),
        'center_velocity': (vx_center, vz_center),
        'idx_center': idx_center,
        'distances': distances
    }

    print(f"\n结果:")
    print(f"  薄膜中心位置: ({x_pos[idx_center]*1e9:.2f}, {z_pos[idx_center]*1e9:.2f}) nm")
    print(f"  薄膜中心速度: vx={vx_center:.3e} m/s, vz={vz_center:.3e} m/s")
    print(f"  瞬时偏转角 α_sample = {alpha_sample_deg:.4f}°")

    return t_array, trajectory, info

def plot_trajectory(t_array, trajectory, info, save_filename='electron_trajectory_single.png'):
    """
    绘制电子轨迹

    参数:
    - t_array: 时间数组
    - trajectory: 轨迹数组
    - info: 信息字典
    - save_filename: 保存文件名
    """
    x_pos = trajectory[:, 0]
    z_pos = trajectory[:, 1]
    vx = trajectory[:, 2]
    vz = trajectory[:, 3]

    theta_foil = info['theta_foil']
    theta_laser = info['theta_laser']
    d = info['foil_thickness']
    nx = info['nx']
    nz = info['nz']
    alpha_sample = info['alpha_sample']
    idx_center = info['idx_center']
    distances = info['distances']

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ========== 主图：轨迹 + 电场背景 ==========
    ax1 = fig.add_subplot(gs[:2, :2])

    # 计算电场分布（作为背景）
    x_grid = np.linspace(-2e-6, 2e-6, 200)
    z_grid = np.linspace(-2e-6, 2e-6, 200)
    Z_grid, X_grid = np.meshgrid(z_grid, x_grid)

    E_field = np.zeros_like(X_grid)
    print("\n生成电场背景...")
    for i in range(0, X_grid.shape[0], 5):  # 降采样加速
        for j in range(0, X_grid.shape[1], 5):
            Ex, Ez = calc_field_at_point(X_grid[i,j], Z_grid[i,j], 0,
                                        theta_foil, theta_laser,
                                        info['wavelength'], info['field_amplitude'],
                                        info['refractive_index'], d)
            E_field[i,j] = np.sqrt(Ex**2 + Ez**2)

    # 绘制电场
    im = ax1.imshow(E_field/info['field_amplitude'],
                    extent=[-2, 2, -2, 2],
                    cmap='RdBu_r', vmin=0, vmax=2,
                    aspect='equal', origin='lower', alpha=0.5)

    # 绘制薄膜
    z_line = np.linspace(-3, 3, 100)
    x_line_front = -z_line * nz / nx  # 前表面
    x_line_back = x_line_front + d * 1e6  # 后表面（粗略）
    ax1.plot(z_line, x_line_front, 'k-', linewidth=4, label='Foil front', zorder=10)
    ax1.plot(z_line, x_line_back, 'k--', linewidth=3, label='Foil back', alpha=0.7, zorder=10)

    # 填充薄膜区域
    ax1.fill_between(z_line, x_line_front, x_line_back,
                     color='gray', alpha=0.2, zorder=9)

    # 绘制法向
    ax1.arrow(0, 0, nx*0.8, nz*0.8, head_width=0.12, head_length=0.1,
             fc='green', ec='green', linewidth=3, zorder=11)
    ax1.text(nx*0.9, nz*0.9, 'Normal', fontsize=12, color='green',
            fontweight='bold', zorder=11)

    # 绘制电子轨迹
    ax1.plot(z_pos*1e6, x_pos*1e6, 'b-', linewidth=3, label='Electron', zorder=12)
    ax1.plot(z_pos[0]*1e6, x_pos[0]*1e6, 'go', markersize=12,
            label='Start', zorder=13)
    ax1.plot(z_pos[idx_center]*1e6, x_pos[idx_center]*1e6, 'r*',
            markersize=20, label='Foil center', zorder=13)
    ax1.plot(z_pos[-1]*1e6, x_pos[-1]*1e6, 'mo', markersize=12,
            label='End', zorder=13)

    # 在薄膜中心画速度矢量
    scale = 0.5
    ax1.arrow(z_pos[idx_center]*1e6, x_pos[idx_center]*1e6,
             vx[idx_center]*scale*1e-6, vz[idx_center]*scale*1e-6,
             head_width=0.08, head_length=0.06, fc='red', ec='red',
             linewidth=2, zorder=13)

    ax1.set_xlabel('z (μm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('x (μm)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Electron Trajectory\nθ_foil={theta_foil}°, θ_laser={theta_laser}°, α_sample={alpha_sample:.4f}°',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    cbar = plt.colorbar(im, ax=ax1, label='|E| / E₀', fraction=0.046, pad=0.04)

    # ========== 速度演化 ==========
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t_array*1e15, vx/1e6, 'r-', linewidth=2, label='vₓ')
    ax2.plot(t_array*1e15, vz/1e6, 'b-', linewidth=2, label='v_z')
    ax2.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='Foil center')
    ax2.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Velocity (10⁶ m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('Velocity vs Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ========== 偏转角演化 ==========
    ax3 = fig.add_subplot(gs[1, 2])
    alpha_t = np.degrees(np.arctan2(vx, vz))
    ax3.plot(t_array*1e15, alpha_t, 'purple', linewidth=2)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='α=0')
    ax3.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='Foil center')
    ax3.plot(t_array[idx_center]*1e15, alpha_sample, 'r*', markersize=15, zorder=10)
    ax3.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('α (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Deflection Angle vs Time\nα_sample={alpha_sample:.4f}°',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ========== 位置演化 ==========
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t_array*1e15, x_pos*1e9, 'r-', linewidth=2, label='x')
    ax4.plot(t_array*1e15, z_pos*1e9, 'b-', linewidth=2, label='z')
    ax4.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label='Foil center')
    ax4.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Position (nm)', fontsize=12, fontweight='bold')
    ax4.set_title('Position vs Time', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ========== 到薄膜的距离 ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t_array*1e15, distances*1e9, 'darkblue', linewidth=2)
    ax5.axhline(0, color='black', linestyle='-', linewidth=2, label='Foil front')
    ax5.axhline(d*1e9, color='black', linestyle='--', linewidth=2, label='Foil back')
    ax5.axhline(d*1e9/2, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='Foil center')
    ax5.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
               linewidth=2, alpha=0.7)
    ax5.fill_between(t_array*1e15, 0, d*1e9, color='gray', alpha=0.2)
    ax5.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Distance to foil (nm)', fontsize=12, fontweight='bold')
    ax5.set_title('Distance to Foil vs Time', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ========== 参数信息文本框 ==========
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    info_text = f"""
SIMULATION PARAMETERS
{'='*35}

Geometry:
  Foil angle:    θ_foil  = {theta_foil:.2f}°
  Laser angle:   θ_laser = {theta_laser:.2f}°

Physical parameters:
  Wavelength:    λ = {info['wavelength']*1e9:.1f} nm
  Foil thickness: d = {d*1e9:.1f} nm
  Refractive index: n = {info['refractive_index']:.2f}
  Field amplitude: E₀ = {info['field_amplitude']:.2e} V/m

Electron:
  Energy:  E = {info['electron_energy_eV']} eV
  Speed:   v = {info['v0']:.3e} m/s
           v/c = {info['v0']/c:.4f}

RESULTS
{'='*35}

At foil center:
  Position: ({info['center_position'][0]*1e9:.2f},
             {info['center_position'][1]*1e9:.2f}) nm
  Velocity: vₓ = {info['center_velocity'][0]:.3e} m/s
           v_z = {info['center_velocity'][1]:.3e} m/s

  ★ α_sample = {alpha_sample:.4f}°

  → {'MAGIC ANGLE!' if abs(alpha_sample) < 0.01 else 'Near magic angle' if abs(alpha_sample) < 0.1 else ''}
    """

    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 轨迹图已保存到: {save_filename}")

    return fig

def main():
    """
    主函数 - 可以在这里设置参数
    """
    print("="*70)
    print("电子轨迹可视化程序")
    print("="*70)

    # ========== 在这里设置参数 ==========

    # 使用扫描得到的最佳magic angle条件
    theta_foil = 80   # 薄膜角度
    theta_laser = 10  # 激光角度

    # 或者手动设置其他参数
    # theta_foil = 45.0
    # theta_laser = 75.0

    electron_energy_eV = 1000
    wavelength = DEFAULT_WAVELENGTH
    foil_thickness = DEFAULT_d

    # =====================================

    # 模拟轨迹
    t_array, trajectory, info = simulate_electron_trajectory(
        theta_foil=theta_foil,
        theta_laser=theta_laser,
        wavelength=wavelength,
        foil_thickness=foil_thickness,
        electron_energy_eV=electron_energy_eV
    )

    # 绘制轨迹
    plot_trajectory(t_array, trajectory, info)

    print("\n" + "="*70)
    print("完成!")
    print("="*70)

    return t_array, trajectory, info

if __name__ == "__main__":
    t_array, trajectory, info = main()

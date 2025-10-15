#!/usr/bin/env python3
"""
简化的电子轨迹可视化程序 - 支持命令行参数

用法:
    python3 plot_trajectory.py 47 62           # 使用指定角度
    python3 plot_trajectory.py                 # 使用默认magic angle
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 物理常数
c = 299792458.0
e = 1.602e-19
m_e = 9.109e-31

# 默认参数
DEFAULT_WAVELENGTH = 1030e-9
DEFAULT_E0 = 1e9
DEFAULT_n = 3.6
DEFAULT_d = 60e-9

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
    """计算空间点(x,z)在时刻t的电场"""
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)
    omega = 2 * np.pi * c / wavelength
    k0 = omega / c
    kx_in = k0 * np.sin(theta_l)
    kz_in = k0 * np.cos(theta_l)
    k_in = np.array([kx_in, kz_in])
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)
    n_foil = np.array([nx, nz])
    cos_theta_in = abs(np.dot(k_in, n_foil)) / k0
    theta_in = np.arccos(np.clip(cos_theta_in, 0, 1))
    r12, t12 = fresnel_p(theta_in, n)
    sin_t = np.sin(theta_in) / n
    cos_t = np.sqrt(1 - sin_t**2)
    theta_t = np.arcsin(sin_t)
    r21, t21 = fresnel_p(theta_t, 1/n)
    phi = 2 * n * k0 * d * cos_t
    denom = 1 - r21**2 * np.exp(1j * phi)
    r_eff = r12 + t12 * t21 * r21 * np.exp(1j * phi) / denom
    t_eff = t12 * t21 * np.exp(1j * phi/2) / denom
    dist = x * nx + z * nz
    if dist < -1e-10:
        phase_in = kx_in * x + kz_in * z - omega * t
        E_in = E0 * np.exp(1j * phase_in)
        k_dot_n = np.dot(k_in, n_foil)
        k_ref = k_in - 2 * k_dot_n * n_foil
        kx_ref = k_ref[0]
        kz_ref = k_ref[1]
        phase_ref = kx_ref * x + kz_ref * z - omega * t
        E_ref = r_eff * E0 * np.exp(1j * phase_ref)
        E_complex = E_in + E_ref
    elif dist > d + 1e-10:
        phase_trans = kx_in * x + kz_in * z - omega * t
        E_complex = t_eff * E0 * np.exp(1j * phase_trans)
    else:
        k_membrane = n * k0
        phase_mem = k_membrane * (kx_in/k0 * x + kz_in/k0 * z) - omega * t
        E_complex = t12 * E0 * np.exp(1j * phase_mem)
    E_magnitude = np.real(E_complex)
    Ex = E_magnitude * np.cos(theta_l)
    Ez = -E_magnitude * np.sin(theta_l)
    return Ex, Ez

def electron_equation(state, t, theta_foil, theta_laser, wavelength, E0, n, d):
    """电子运动方程"""
    x, z, vx, vz = state
    Ex, Ez = calc_field_at_point(x, z, t, theta_foil, theta_laser, wavelength, E0, n, d)
    ax = (e / m_e) * Ex
    az = (e / m_e) * Ez
    return [vx, vz, ax, az]

def simulate_and_plot(theta_foil, theta_laser,
                     electron_energy_eV=1000,
                     wavelength=DEFAULT_WAVELENGTH,
                     d=DEFAULT_d,
                     E0=DEFAULT_E0,
                     n=DEFAULT_n):
    """模拟并绘制电子轨迹"""

    print(f"\n{'='*70}")
    print(f"模拟参数: θ_foil={theta_foil}°, θ_laser={theta_laser}°")
    print(f"{'='*70}")

    # 电子初始条件
    E_k = electron_energy_eV * e
    v0 = np.sqrt(2 * E_k / m_e)
    theta_f = np.radians(theta_foil)
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)

    x0 = 0.0
    z0 = -2e-6
    vx0 = 0.0
    vz0 = v0
    initial_state = [x0, z0, vx0, vz0]

    # 计算模拟时间
    initial_dist_to_foil = abs(x0 * nx + z0 * nz)
    time_estimate = initial_dist_to_foil / v0 * 2.5
    t_max = max(2.5e-13, time_estimate)
    t_array = np.linspace(0, t_max, 20000)

    print(f"求解运动方程...")
    trajectory = odeint(electron_equation, initial_state, t_array,
                       args=(theta_foil, theta_laser, wavelength, E0, n, d))

    x_pos = trajectory[:, 0]
    z_pos = trajectory[:, 1]
    vx = trajectory[:, 2]
    vz = trajectory[:, 3]

    # 找薄膜中心
    distances = x_pos * nx + z_pos * nz
    idx_center = np.argmin(np.abs(distances - d/2))

    # 计算瞬时偏转角
    vx_center = vx[idx_center]
    vz_center = vz[idx_center]
    alpha_sample = np.degrees(np.arctan2(vx_center, vz_center))

    print(f"\n结果:")
    print(f"  薄膜中心: ({x_pos[idx_center]*1e9:.2f}, {z_pos[idx_center]*1e9:.2f}) nm")
    print(f"  薄膜中心速度: vx={vx_center:.3e} m/s, vz={vz_center:.3e} m/s")
    print(f"  ★ α_sample = {alpha_sample:.6f}°")

    if abs(alpha_sample) < 0.01:
        print(f"  → MAGIC ANGLE!")
    elif abs(alpha_sample) < 0.1:
        print(f"  → 接近magic angle")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 轨迹图
    ax = axes[0, 0]
    ax.plot(z_pos*1e6, x_pos*1e6, 'b-', linewidth=2, label='Electron')
    ax.plot(z_pos[0]*1e6, x_pos[0]*1e6, 'go', markersize=10, label='Start')
    ax.plot(z_pos[idx_center]*1e6, x_pos[idx_center]*1e6, 'r*',
           markersize=20, label='Foil center')
    ax.plot(z_pos[-1]*1e6, x_pos[-1]*1e6, 'mo', markersize=10, label='End')

    # 画薄膜
    z_line = np.linspace(-3, 3, 100)
    x_line = -z_line * nz / nx
    ax.plot(z_line, x_line, 'k-', linewidth=4, label='Foil', alpha=0.7)

    # 法向
    ax.arrow(0, 0, nx*0.8, nz*0.8, head_width=0.12, head_length=0.1,
            fc='green', ec='green', linewidth=3)
    ax.text(nx*0.9, nz*0.9, 'Normal', fontsize=12, color='green', fontweight='bold')

    # 速度矢量
    scale = 0.5
    ax.arrow(z_pos[idx_center]*1e6, x_pos[idx_center]*1e6,
            vx[idx_center]*scale*1e-6, vz[idx_center]*scale*1e-6,
            head_width=0.08, head_length=0.06, fc='red', ec='red', linewidth=2)

    ax.set_xlabel('z (μm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('x (μm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Electron Trajectory\nθ_foil={theta_foil}°, θ_laser={theta_laser}°',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # 2. 速度演化
    ax = axes[0, 1]
    ax.plot(t_array*1e15, vx/1e6, 'r-', linewidth=2, label='vₓ')
    ax.plot(t_array*1e15, vz/1e6, 'b-', linewidth=2, label='v_z')
    ax.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
              linewidth=2, alpha=0.7, label='Foil center')
    ax.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Velocity (10⁶ m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Velocity vs Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 偏转角演化
    ax = axes[1, 0]
    alpha_t = np.degrees(np.arctan2(vx, vz))
    ax.plot(t_array*1e15, alpha_t, 'purple', linewidth=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='α=0')
    ax.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
              linewidth=2, alpha=0.7, label='Foil center')
    ax.plot(t_array[idx_center]*1e15, alpha_sample, 'r*', markersize=15)
    ax.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('α (degrees)', fontsize=12, fontweight='bold')
    ax.set_title(f'Deflection Angle vs Time\nα_sample={alpha_sample:.6f}°',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 到薄膜距离
    ax = axes[1, 1]
    ax.plot(t_array*1e15, distances*1e9, 'darkblue', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=2, label='Foil front')
    ax.axhline(d*1e9, color='black', linestyle='--', linewidth=2, label='Foil back')
    ax.axhline(d*1e9/2, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label='Foil center')
    ax.axvline(t_array[idx_center]*1e15, color='green', linestyle='--',
              linewidth=2, alpha=0.7)
    ax.fill_between(t_array*1e15, 0, d*1e9, color='gray', alpha=0.2)
    ax.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance to foil (nm)', fontsize=12, fontweight='bold')
    ax.set_title('Distance to Foil vs Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'trajectory_foil{theta_foil:.0f}_laser{theta_laser:.0f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图像已保存: {filename}")

    return alpha_sample

def main():
    """主函数 - 支持命令行参数"""

    print("="*70)
    print("电子轨迹可视化 - 命令行版本")
    print("="*70)

    # 解析命令行参数
    if len(sys.argv) == 3:
        try:
            theta_foil = float(sys.argv[1])
            theta_laser = float(sys.argv[2])
            print(f"使用命令行参数: θ_foil={theta_foil}°, θ_laser={theta_laser}°")
        except ValueError:
            print("错误：参数必须是数字")
            print(f"用法: {sys.argv[0]} <theta_foil> <theta_laser>")
            print(f"例如: {sys.argv[0]} 47 62")
            sys.exit(1)
    elif len(sys.argv) == 1:
        # 默认使用最佳magic angle
        theta_foil = 47.0
        theta_laser = 62.0
        print(f"使用默认magic angle: θ_foil={theta_foil}°, θ_laser={theta_laser}°")
    else:
        print(f"用法: {sys.argv[0]} [theta_foil theta_laser]")
        print(f"例如: {sys.argv[0]} 47 62")
        print(f"或者: {sys.argv[0]}  (使用默认参数)")
        sys.exit(1)

    # 运行模拟
    alpha_sample = simulate_and_plot(theta_foil, theta_laser)

    print("\n" + "="*70)
    print("完成!")
    print("="*70)

    return alpha_sample

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar

# 物理常数
c = 299792458.0  # 光速
e = 1.602e-19    # 电子电荷
m_e = 9.109e-31  # 电子质量
WAVELENGTH = 1030e-9
E0 = 1e9
n = 3.6
d = 60e-9

# 薄膜和激光配置
theta_foil = 45    # 薄膜法向角度
theta_laser = 75   # 激光入射角度

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

def calc_field_at_point(x, z, t):
    """
    计算空间点(x,z)在时刻t的电场
    返回: Ex, Ez (实数值)
    """
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)

    omega = 2 * np.pi * c / WAVELENGTH
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

    # 判断区域
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

    # 电场是p偏振，需要计算矢量分量
    # p偏振：电场在入射平面内
    E_magnitude = np.real(E_complex)

    # 电场方向（简化：假设电场主要沿激光传播方向的垂直方向）
    # 对于p偏振，电场在入射平面内
    Ex = E_magnitude * np.cos(theta_l)  # 近似
    Ez = -E_magnitude * np.sin(theta_l)

    return Ex, Ez

def electron_equation(state, t, electron_energy_eV):
    """
    电子运动方程: dv/dt = (e/m)*E(x,t)
    state = [x, z, vx, vz]
    """
    x, z, vx, vz = state

    # 计算当前位置的电场
    Ex, Ez = calc_field_at_point(x, z, t)

    # 加速度
    ax = (e / m_e) * Ex
    az = (e / m_e) * Ez

    return [vx, vz, ax, az]

def simulate_electron(x0, z0, v0, angle_deg, electron_energy_eV, t_max=1e-13, n_steps=10000):
    """
    模拟电子轨迹

    参数:
    - x0, z0: 初始位置 (m)
    - v0: 初始速度大小 (m/s)
    - angle_deg: 初始速度方向（相对于z轴，度）
    - electron_energy_eV: 电子能量(eV)，用于相对论修正
    - t_max: 模拟时间
    - n_steps: 时间步数

    返回:
    - t_array: 时间数组
    - trajectory: [x, z, vx, vz] 数组
    """
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.sin(angle_rad)
    vz0 = v0 * np.cos(angle_rad)

    initial_state = [x0, z0, vx0, vz0]
    t_array = np.linspace(0, t_max, n_steps)

    # 求解ODE
    trajectory = odeint(electron_equation, initial_state, t_array, args=(electron_energy_eV,))

    return t_array, trajectory

def find_membrane_crossing_angle(trajectory, x_positions, z_positions):
    """
    计算电子穿过薄膜中心时的入射角度

    返回:
    - angle: 相对于薄膜法向的入射角度（度）
    """
    theta_f = np.radians(theta_foil)
    nx = np.sin(theta_f)
    nz = np.cos(theta_f)

    # 薄膜中心位置（沿法向距离为d/2）
    membrane_center_dist = d / 2

    # 找到最接近薄膜中心的点
    distances = x_positions * nx + z_positions * nz
    idx = np.argmin(np.abs(distances - membrane_center_dist))

    # 在该点的速度
    vx = trajectory[idx, 2]
    vz = trajectory[idx, 3]
    v_vec = np.array([vx, vz])

    # 速度与法向的夹角
    v_magnitude = np.sqrt(vx**2 + vz**2)
    if v_magnitude < 1e-10:
        return 90.0

    cos_angle = np.dot(v_vec, [nx, nz]) / v_magnitude
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))

    # 转换为度并相对于法向
    angle_deg = np.degrees(angle_rad)

    # 返回相对于法向的角度（0度表示垂直入射）
    return 90 - angle_deg if angle_deg > 90 else angle_deg

def optimize_initial_conditions(electron_energy_eV=1000, fine_tune=True):
    """
    优化初始条件，使电子以接近0度的角度入射薄膜

    参数:
    - electron_energy_eV: 电子能量(eV)
    - fine_tune: 是否进行精细优化

    返回:
    - 最优初始位置和角度
    """
    # 根据能量计算初始速度
    # E_k = 0.5 * m * v^2
    E_k = electron_energy_eV * e  # 转换为焦耳
    v0 = np.sqrt(2 * E_k / m_e)

    print(f"\n电子能量: {electron_energy_eV} eV")
    print(f"初始速度: {v0:.3e} m/s ({v0/c:.4f}c)")

    best_results = []

    # 第一轮：粗扫描
    print("\n第一轮：粗扫描参数空间...")
    x0_range = np.linspace(-1.5e-6, -0.1e-6, 12)
    z0_range = np.linspace(-1.5e-6, 1.5e-6, 12)
    angle_range = np.linspace(50, 90, 20)

    for i, x0 in enumerate(x0_range):
        for z0 in z0_range:
            for angle in angle_range:
                try:
                    t_array, trajectory = simulate_electron(
                        x0, z0, v0, angle, electron_energy_eV,
                        t_max=5e-14, n_steps=5000
                    )

                    x_pos = trajectory[:, 0]
                    z_pos = trajectory[:, 1]

                    # 检查是否穿过薄膜
                    theta_f = np.radians(theta_foil)
                    nx = np.sin(theta_f)
                    nz = np.cos(theta_f)
                    distances = x_pos * nx + z_pos * nz

                    # 如果穿过薄膜区域
                    if np.any((distances > 0) & (distances < d)):
                        crossing_angle = find_membrane_crossing_angle(trajectory, x_pos, z_pos)
                        best_results.append({
                            'x0': x0,
                            'z0': z0,
                            'angle': angle,
                            'crossing_angle': abs(crossing_angle),
                            'trajectory': trajectory,
                            't_array': t_array
                        })
                except:
                    continue

        if (i+1) % 3 == 0:
            print(f"  进度: {(i+1)/len(x0_range)*100:.0f}%")

    if not best_results:
        print("未找到穿过薄膜的轨迹！")
        return None

    # 找到入射角最小的
    best_results.sort(key=lambda x: x['crossing_angle'])

    print(f"\n第一轮找到 {len(best_results)} 个有效轨迹")

    # 第二轮：精细优化
    if fine_tune and len(best_results) > 0:
        print("\n第二轮：精细优化最佳区域...")

        # 围绕最佳结果进行精细搜索
        best = best_results[0]
        x0_fine = np.linspace(best['x0'] - 0.2e-6, best['x0'] + 0.2e-6, 15)
        z0_fine = np.linspace(best['z0'] - 0.2e-6, best['z0'] + 0.2e-6, 15)
        angle_fine = np.linspace(max(best['angle'] - 10, 40),
                                  min(best['angle'] + 10, 90), 20)

        fine_results = []
        for x0 in x0_fine:
            for z0 in z0_fine:
                for angle in angle_fine:
                    try:
                        t_array, trajectory = simulate_electron(
                            x0, z0, v0, angle, electron_energy_eV,
                            t_max=5e-14, n_steps=8000
                        )

                        x_pos = trajectory[:, 0]
                        z_pos = trajectory[:, 1]

                        theta_f = np.radians(theta_foil)
                        nx = np.sin(theta_f)
                        nz = np.cos(theta_f)
                        distances = x_pos * nx + z_pos * nz

                        if np.any((distances > 0) & (distances < d)):
                            crossing_angle = find_membrane_crossing_angle(trajectory, x_pos, z_pos)
                            fine_results.append({
                                'x0': x0,
                                'z0': z0,
                                'angle': angle,
                                'crossing_angle': abs(crossing_angle),
                                'trajectory': trajectory,
                                't_array': t_array
                            })
                    except:
                        continue

        if fine_results:
            fine_results.sort(key=lambda x: x['crossing_angle'])
            # 合并结果
            all_results = fine_results + best_results
            all_results.sort(key=lambda x: x['crossing_angle'])
            best_results = all_results

            print(f"第二轮找到 {len(fine_results)} 个精细轨迹")

    print(f"\n总共找到 {len(best_results)} 个有效轨迹")
    print(f"\n最优结果（前5个）:")
    for i, result in enumerate(best_results[:5]):
        print(f"  {i+1}. 初始: x0={result['x0']*1e6:.3f}μm, z0={result['z0']*1e6:.3f}μm, "
              f"发射角={result['angle']:.1f}°")
        print(f"     入射角: {result['crossing_angle']:.2f}°")

    return best_results

def plot_best_trajectories(best_results, electron_energy_eV, n_plot=5):
    """
    绘制最优轨迹
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：电场背景 + 轨迹
    ax = axes[0]

    # 绘制电场分布（背景）
    x = np.linspace(-2e-6, 2e-6, 300)
    z = np.linspace(-2e-6, 2e-6, 300)
    Z, X = np.meshgrid(z, x)

    E_field = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Ex, Ez = calc_field_at_point(X[i,j], Z[i,j], 0)
            E_field[i,j] = np.sqrt(Ex**2 + Ez**2)

    E_norm = E_field / E0
    im = ax.imshow(E_norm, extent=[-2, 2, -2, 2],
                   cmap='RdBu_r', vmin=0, vmax=2,
                   aspect='equal', origin='lower', alpha=0.6)

    # 画薄膜
    theta_f_rad = np.radians(theta_foil)
    z_line = np.linspace(-3, 3, 100)
    x_line = -z_line * np.cos(theta_f_rad) / np.sin(theta_f_rad)
    ax.plot(z_line, x_line, 'k-', linewidth=3, alpha=0.9, label='Foil')

    # 画薄膜法向
    z_normal_start = 0
    x_normal_start = 0
    z_normal_end = z_normal_start + 0.5 * np.cos(theta_f_rad)
    x_normal_end = x_normal_start + 0.5 * np.sin(theta_f_rad)
    ax.arrow(z_normal_start, x_normal_start,
             z_normal_end - z_normal_start, x_normal_end - x_normal_start,
             head_width=0.1, head_length=0.08, fc='green', ec='green',
             linewidth=2, label='Normal')

    # 绘制轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, n_plot))
    for i, result in enumerate(best_results[:n_plot]):
        traj = result['trajectory']
        x_pos = traj[:, 0] * 1e6
        z_pos = traj[:, 1] * 1e6
        ax.plot(z_pos, x_pos, '-', linewidth=2, color=colors[i],
                label=f"#{i+1}: θ={result['crossing_angle']:.2f}°")
        ax.plot(z_pos[0], x_pos[0], 'o', markersize=8, color=colors[i])

    ax.set_xlabel('z (μm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('x (μm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Electron Trajectories (E={electron_energy_eV} eV)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='|E| / E₀', fraction=0.046, pad=0.04)

    # 右图：入射角统计
    ax = axes[1]
    angles = [r['crossing_angle'] for r in best_results[:20]]
    colors_bar = [colors[i] if i < n_plot else 'gray' for i in range(len(angles))]
    bars = ax.bar(range(len(angles)), angles, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target (0°)')
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='5° threshold')
    ax.set_xlabel('Trajectory Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Incident Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('Incident Angles at Membrane Center', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('electron_trajectory_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def main():
    """主函数"""
    print("="*60)
    print("电子在电磁场中的轨迹模拟与入射角度优化")
    print("="*60)

    # 电子能量（可调整）
    electron_energy_eV = 1000  # 1 keV

    # 优化初始条件
    best_results = optimize_initial_conditions(electron_energy_eV)

    if best_results:
        # 绘制结果
        plot_best_trajectories(best_results, electron_energy_eV, n_plot=5)

        # 输出最优结果详细信息
        best = best_results[0]
        print(f"\n" + "="*60)
        print(f"最优配置:")
        print(f"="*60)
        print(f"初始位置: x0 = {best['x0']*1e6:.3f} μm, z0 = {best['z0']*1e6:.3f} μm")
        print(f"初始发射角: {best['angle']:.1f}° (相对于z轴)")
        print(f"薄膜中心入射角: {best['crossing_angle']:.3f}° (相对于法向)")
        print(f"\n结论: 入射角度 {'非常接近' if best['crossing_angle'] < 5 else '较为接近'} 0°")
        print("="*60)

    return best_results

if __name__ == "__main__":
    results = main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 物理常数
c = 299792458.0
e = 1.602e-19
m_e = 9.109e-31

# 测试参数
theta_foil = 45  # 薄膜角度
theta_laser = 75  # 激光角度
electron_energy_eV = 1000

# 电子速度
E_k = electron_energy_eV * e
v0 = np.sqrt(2 * E_k / m_e)

print(f"电子能量: {electron_energy_eV} eV")
print(f"电子速度: {v0:.3e} m/s ({v0/c:.4f}c)")

# 薄膜法向
theta_f = np.radians(theta_foil)
nx = np.sin(theta_f)
nz = np.cos(theta_f)

print(f"\n薄膜倾角: {theta_foil}°")
print(f"薄膜法向: ({nx:.3f}, {nz:.3f})")

# 测试：电子从z轴负方向沿+z方向入射
x0 = 0.0
z0 = -2e-6  # 从-2μm处开始

print(f"\n初始位置: ({x0*1e6:.2f}, {z0*1e6:.2f}) μm")
print(f"初始速度: vx=0, vz={v0:.3e} m/s (+z方向)")

# 模拟无电场情况下的自由飞行
t_max = 1e-13
t_array = np.linspace(0, t_max, 1000)

# 自由飞行
x_traj = x0 * np.ones_like(t_array)
z_traj = z0 + v0 * t_array

# 计算到薄膜的距离
# 薄膜前表面在原点，法向为(nx, nz)
distances = x_traj * nx + z_traj * nz

print(f"\n时间演化:")
print(f"t=0: 距离薄膜 {distances[0]*1e9:.2f} nm")
print(f"t={t_max*1e15:.1f} fs: 距离薄膜 {distances[-1]*1e9:.2f} nm")
print(f"最终z位置: {z_traj[-1]*1e6:.2f} μm")

# 检查是否穿过薄膜
d = 60e-9
crossed = np.any((distances > 0) & (distances < d))
print(f"\n是否穿过薄膜（0到{d*1e9:.0f} nm）: {crossed}")

if not crossed:
    # 计算需要的时间
    # 电子需要到达薄膜前表面: dist = 0
    # x*nx + z*nz = 0
    # 0*nx + (z0 + v0*t)*nz = 0
    # t = -z0*nz / (v0*nz) = -z0/v0  (因为电子沿+z方向，nz>0)

    # 更正确的计算：
    # 需要距离 = 0 - (x0*nx + z0*nz) = -(x0*nx + z0*nz)
    # 速度沿+z方向的分量对距离的贡献 = v0*nz
    initial_dist = x0*nx + z0*nz
    print(f"初始距离: {initial_dist*1e9:.2f} nm")
    print(f"需要穿越距离: {-initial_dist*1e9:.2f} nm")

    time_to_foil = -initial_dist / (v0 * nz)
    print(f"到达薄膜所需时间: {time_to_foil*1e15:.2f} fs")
    print(f"当前模拟时间: {t_max*1e15:.1f} fs")

    if time_to_foil > t_max:
        print(f"\n问题：模拟时间太短！需要至少 {time_to_foil*1e15:.1f} fs")

# 可视化
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制轨迹
ax.plot(z_traj*1e6, x_traj*1e6, 'b-', linewidth=2, label='Electron trajectory')
ax.plot(z0*1e6, x0*1e6, 'go', markersize=10, label='Start')
ax.plot(z_traj[-1]*1e6, x_traj[-1]*1e6, 'ro', markersize=10, label='End')

# 绘制薄膜
z_line = np.linspace(-3, 3, 100)
x_line = -z_line * nz / nx  # 薄膜前表面经过原点
ax.plot(z_line, x_line, 'k-', linewidth=3, label='Foil front')

# 薄膜后表面
x_line_back = x_line + d * nx / np.sqrt(nx**2 + nz**2) * 1e6
ax.plot(z_line, x_line_back, 'k--', linewidth=2, label='Foil back')

# 法向箭头
ax.arrow(0, 0, nx*0.5, nz*0.5, head_width=0.1, head_length=0.1,
         fc='green', ec='green', linewidth=2)
ax.text(nx*0.6, nz*0.6, 'Normal', fontsize=12, color='green')

ax.set_xlabel('z (μm)', fontsize=13)
ax.set_ylabel('x (μm)', fontsize=13)
ax.set_title(f'Free Electron Trajectory (no E-field)\nθ_foil={theta_foil}°, E={electron_energy_eV}eV',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig('debug_trajectory.png', dpi=200)
print("\n可视化已保存到: debug_trajectory.png")
plt.show()

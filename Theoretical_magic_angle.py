import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from tqdm import tqdm  # 进度条

# 物理常数
c = 299792458.0           # 光速 m/s
e = 1.602176634e-19       # 电子电荷 C
m_e = 9.1093837015e-31    # 电子质量 kg
hbar = 1.054571817e-34    # 约化普朗克常数

# 电子参数
E_kin = 70e3 * e          # 70 keV
gamma = 1 + E_kin / (m_e * c**2)  # 洛伦兹因子
v0 = c * np.sqrt(1 - 1/gamma**2)  # 电子速度
p0 = gamma * m_e * v0     # 电子动量

print(f"电子参数：")
print(f"  动能: {E_kin/e/1e3:.1f} keV")
print(f"  速度: {v0/c:.3f}c")
print(f"  洛伦兹因子: {gamma:.3f}")
print(f"  动量: {p0:.3e} kg·m/s")

def fresnel_coefficients(theta_in, n, polarization='p'):
    """
    计算 Fresnel 反射和透射系数
    
    参数:
        theta_in: 入射角（弧度）
        n: 折射率
        polarization: 'p' 或 's'
    
    返回:
        r: 反射系数
        t: 透射系数
    """
    cos_theta_in = np.cos(theta_in)
    
    # Snell 定律
    sin_theta_t = np.sin(theta_in) / n
    
    # 检查全反射
    if np.abs(sin_theta_t) > 1:
        return -1.0, 0.0  # 全反射
    
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    
    if polarization == 'p':
        # p 偏振（TM）
        r = (n * cos_theta_in - cos_theta_t) / (n * cos_theta_in + cos_theta_t)
        t = 2 * cos_theta_in / (n * cos_theta_in + cos_theta_t)
    else:
        # s 偏振（TE）
        r = (cos_theta_in - n * cos_theta_t) / (cos_theta_in + n * cos_theta_t)
        t = 2 * cos_theta_in / (cos_theta_in + n * cos_theta_t)
    
    return r, t

# 测试
theta = np.radians(45)
r_p, t_p = fresnel_coefficients(theta, n=3.6, polarization='p')
print(f"\nFresnel 系数测试（45°, Si）：")
print(f"  r_p = {r_p:.3f}")
print(f"  t_p = {t_p:.3f}")
print(f"  |r|² + |t|²/n² = {np.abs(r_p)**2 + np.abs(t_p)**2/3.6**2:.3f}")  # 能量守恒

def em_field_before_membrane(z, t, theta_foil, theta_laser, 
                             wavelength, E0, n, d):
    """
    计算薄膜前（z < 0）的电磁场
    
    参数:
        z: 位置（沿电子轨迹）
        t: 时间
        theta_foil: 薄膜角度（度）
        theta_laser: 激光角度（度）
        wavelength: 波长（m）
        E0: 入射场强（V/m）
        n: 折射率
        d: 薄膜厚度（m）
    
    返回:
        E: 电场 [Ex, Ey, Ez]
        B: 磁场 [Bx, By, Bz]
    """
    # 转换为弧度
    theta_f = np.radians(theta_foil)
    theta_l = np.radians(theta_laser)
    
    omega = 2 * np.pi * c / wavelength
    k0 = omega / c
    
    # 入射波波矢（在 xz 平面）
    k_in = k0 * np.array([
        np.sin(theta_l),
        0,
        np.cos(theta_l)
    ])
    
    # 入射角（相对于薄膜法向）
    n_foil = np.array([np.sin(theta_f), 0, np.cos(theta_f)])
    theta_in = np.arccos(np.abs(np.dot(k_in/k0, n_foil)))
    
    # Fresnel 系数
    r1, t1 = fresnel_coefficients(theta_in, n, 'p')
    r2, t2 = fresnel_coefficients(np.arcsin(np.sin(theta_in)/n), n, 'p')
    
    # 简化：只考虑入射波 + 一次反射
    # （完整版需要无穷级数）
    
    # 入射波
    phase_in = np.dot(k_in, [0, 0, z]) - omega * t
    E_in = E0 * np.array([
        np.cos(theta_l),  # Ex
        0,                # Ey (p-pol)
        -np.sin(theta_l)  # Ez
    ]) * np.cos(phase_in)
    
    # 反射波
    k_ref = k0 * np.array([
        np.sin(theta_l),
        0,
        -np.cos(theta_l)  # 反射后 z 分量反向
    ])
    
    phase_ref = np.dot(k_ref, [0, 0, z]) - omega * t
    E_ref = r1 * E0 * np.array([
        np.cos(theta_l),
        0,
        np.sin(theta_l)  # 注意符号
    ]) * np.cos(phase_ref)
    
    # 总场
    E = E_in + E_ref
    
    # 磁场（简化）：B = k × E / omega
    B_in = np.cross(k_in, E_in) / omega
    B_ref = np.cross(k_ref, E_ref) / omega
    B = B_in + B_ref
    
    return E, B

# 测试
E, B = em_field_before_membrane(
    z=-1e-6,  # -1 μm
    t=0,
    theta_foil=35,
    theta_laser=155,
    wavelength=1030e-9,
    E0=1e9,
    n=3.6,
    d=60e-9
)

print(f"\n电磁场测试：")
print(f"  E = [{E[0]:.2e}, {E[1]:.2e}, {E[2]:.2e}] V/m")
print(f"  B = [{B[0]:.2e}, {B[1]:.2e}, {B[2]:.2e}] T")

def calculate_alpha_sample(theta_foil, theta_laser, 
                          wavelength=1030e-9, E0=1e9, 
                          n=3.6, d=60e-9):
    """
    计算 α_sample
    
    返回:
        alpha_sample: 偏转角（弧度）
        alpha_x, alpha_y: 各分量
    """
    omega = 2 * np.pi * c / wavelength
    T = 2 * np.pi / omega  # 周期
    
    # 时间数组（从 -10T 到 0）
    N_points = 500
    t_array = np.linspace(-10*T, 0, N_points)
    dt = t_array[1] - t_array[0]
    
    # 初始化动量变化
    dp_x = 0.0
    dp_y = 0.0
    
    # 电子速度向量
    v_vec = np.array([0, 0, v0])
    
    # 沿轨迹积分
    for t in t_array:
        z = v0 * t  # 电子位置
        
        # 电磁场
        E, B = em_field_before_membrane(
            z, t, theta_foil, theta_laser, 
            wavelength, E0, n, d
        )
        
        # Lorentz 力
        F = -e * (E + np.cross(v_vec, B))
        
        # 累积动量
        dp_x += F[0] * dt
        dp_y += F[1] * dt
    
    # 转换为角度
    alpha_x = dp_x / p0
    alpha_y = dp_y / p0
    alpha_sample = np.sqrt(alpha_x**2 + alpha_y**2)
    
    return alpha_sample, alpha_x, alpha_y

# 测试单点
alpha, alpha_x, alpha_y = calculate_alpha_sample(
    theta_foil=35,
    theta_laser=155
)

print(f"\n单点测试：")
print(f"  θ_foil = 35°, θ_laser = 155°")
print(f"  α_sample = {alpha*1e3:.3f} mrad")
print(f"  α_x = {alpha_x*1e3:.3f} mrad")
print(f"  α_y = {alpha_y*1e3:.3f} mrad")

def scan_parameter_space(theta_foil_range, theta_laser_range,
                         wavelength=1030e-9, E0=1e9, 
                         n=3.6, d=60e-9):
    """
    扫描参数空间，寻找魔角
    
    返回:
        alpha_map: 2D 数组
        theta_foil_array, theta_laser_array
    """
    Nf = len(theta_foil_range)
    Nl = len(theta_laser_range)
    
    alpha_map = np.zeros((Nf, Nl))
    
    print("扫描参数空间...")
    
    # 使用 tqdm 显示进度
    from tqdm import tqdm
    
    for i, theta_f in enumerate(tqdm(theta_foil_range)):
        for j, theta_l in enumerate(theta_laser_range):
            
            try:
                alpha, _, _ = calculate_alpha_sample(
                    theta_f, theta_l, 
                    wavelength, E0, n, d
                )
                alpha_map[i, j] = alpha
                
            except:
                # 如果计算失败（例如全反射）
                alpha_map[i, j] = np.nan
    
    return alpha_map

# 定义搜索范围
theta_foil_array = np.linspace(-60, 60, 61)   # 2° 步长
theta_laser_array = np.linspace(0, 180, 91)   # 2° 步长

# 扫描（这会需要几分钟）
alpha_map = scan_parameter_space(
    theta_foil_array,
    theta_laser_array,
    wavelength=1030e-9,
    E0=1e9,
    n=3.6,
    d=60e-9
)

print(f"\n扫描完成！")
print(f"  α_sample 范围: [{np.nanmin(alpha_map)*1e3:.3f}, {np.nanmax(alpha_map)*1e3:.3f}] mrad")

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
    ax.set_title('α_sample (mrad)', fontsize=14)
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
    plt.savefig('alpha_sample_map.png', dpi=300)
    plt.show()
    
    # 找到所有候选点（α < 阈值）
    threshold = np.nanpercentile(alpha_map, 10)  # 最小的 10%
    
    candidates = []
    for i in range(len(theta_foil_array)):
        for j in range(len(theta_laser_array)):
            if alpha_map[i, j] < threshold:
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
              f"{cand['alpha_sample']*1e3:>13.3f}")
    
    return candidates

# 可视化并找候选点
candidates = visualize_and_find_candidates(
    alpha_map,
    theta_foil_array,
    theta_laser_array
)

def fine_scan_around_candidate(theta_f_center, theta_l_center,
                               d_theta=5, n_points=21,
                               **kwargs):
    """
    在候选点附近精细扫描
    
    参数:
        theta_f_center, theta_l_center: 中心角度
        d_theta: 扫描范围（±）
        n_points: 每个方向的点数
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
    
    print(f"\n精细扫描：")
    print(f"  中心：({theta_f_center:.1f}°, {theta_l_center:.1f}°)")
    print(f"  范围：±{d_theta}°")
    print(f"  步长：{2*d_theta/(n_points-1):.2f}°")
    
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
    print(f"  α_sample = {alpha_opt*1e3:.3f} mrad")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    im = plt.contourf(theta_laser_fine, theta_foil_fine, alpha_map_fine*1e3,
                      levels=20, cmap='RdYlGn_r')
    plt.plot(theta_l_opt, theta_f_opt, 'r*', markersize=20)
    plt.xlabel('θ_laser (degrees)', fontsize=12)
    plt.ylabel('θ_foil (degrees)', fontsize=12)
    plt.title('Fine scan of α_sample (mrad)', fontsize=14)
    plt.colorbar(im, label='α_sample (mrad)')
    plt.tight_layout()
    plt.savefig('alpha_sample_fine_scan.png', dpi=300)
    plt.show()
    
    return theta_f_opt, theta_l_opt, alpha_opt

# 对第一个候选点进行精细扫描
best_candidate = candidates[0]

theta_f_opt, theta_l_opt, alpha_opt = fine_scan_around_candidate(
    theta_f_center=best_candidate['theta_foil'],
    theta_l_center=best_candidate['theta_laser'],
    d_theta=3,        # ±3°
    n_points=31,      # 0.2° 步长
    wavelength=1030e-9,
    E0=1e9,
    n=3.6,
    d=60e-9
)
"""
计算合适的时间快照数值
"""
import numpy as np

# 参数
WAVELENGTH = 1030e-9  # 1030 nm
c = 299792458.0       # 光速 m/s

# 计算关键时间尺度
print("="*60)
print("电磁波时间尺度")
print("="*60)

# 1. 光周期
omega = 2 * np.pi * c / WAVELENGTH
T = 2 * np.pi / omega

print(f"\n波长: {WAVELENGTH*1e9:.0f} nm")
print(f"频率: {omega/(2*np.pi):.3e} Hz")
print(f"角频率 ω: {omega:.3e} rad/s")
print(f"\n光周期 T = 2π/ω:")
print(f"  T = {T:.3e} s")
print(f"  T = {T*1e15:.3f} fs (飞秒)")
print(f"  T = {T*1e18:.3f} as (阿秒)")

# 2. 推荐的时间快照
print("\n" + "="*60)
print("推荐的时间快照值")
print("="*60)

snapshots = [0, T/8, T/4, T/2, 3*T/4, T]
labels = ['0 (初始)', 'T/8', 'T/4 (1/4周期)', 'T/2 (半周期)', '3T/4', 'T (完整周期)']

for t, label in zip(snapshots, labels):
    print(f"\nt_snapshot = {t:.3e} s  ({label})")
    print(f"           = {t*1e15:.6f} fs")
    print(f"           = {t*1e18:.3f} as")
    print(f"  相位: ωt = {omega*t:.3f} rad = {np.degrees(omega*t):.1f}°")

# 3. 观察效果
print("\n" + "="*60)
print("不同时间点的观察效果")
print("="*60)

print("\n时间 t = 0（初始）:")
print("  - 电场处于某个相位")
print("  - 看到静态的干涉图样")

print("\nt = T/4（四分之一周期）:")
print("  - 相位变化 90°")
print("  - 波峰和波谷位置移动 λ/4")
print("  - 干涉条纹向前移动")

print("\nt = T/2（半周期）:")
print("  - 相位变化 180°")
print("  - 波峰变波谷，波谷变波峰")
print("  - 红蓝颜色完全反转")

print("\nt = T（完整周期）:")
print("  - 相位变化 360°")
print("  - 回到初始状态")
print("  - 与 t=0 相同")

# 4. 代码中如何设置
print("\n" + "="*60)
print("在代码中如何设置")
print("="*60)

print("\n方法1：使用秒（科学计数法）")
print(f"  t_snapshot = {T/4:.3e}  # T/4 = {T/4*1e15:.3f} fs")

print("\n方法2：使用飞秒")
print(f"  t_snapshot = {T/4*1e15:.6f}e-15  # {T/4*1e15:.3f} fs")

print("\n方法3：使用阿秒（推荐）")
print(f"  t_snapshot = {T/4*1e18:.3f}e-18  # {T/4*1e18:.3f} as")

print("\n方法4：直接计算")
print(f"  T = 2*np.pi*c/WAVELENGTH / (2*np.pi)  # 光周期")
print(f"  t_snapshot = T/4  # 或 T/2, T/8 等")

# 5. 推荐值
print("\n" + "="*60)
print("✨ 推荐设置")
print("="*60)

print(f"\n对于 λ = {WAVELENGTH*1e9:.0f} nm 的激光：")
print(f"\n选项A：看初始状态")
print(f"  t_snapshot = 0")

print(f"\n选项B：看四分之一周期（推荐，条纹移动明显）")
print(f"  t_snapshot = {T/4*1e15:.6f}e-15  # 飞秒")
print(f"  或")
print(f"  t_snapshot = {T/4*1e18:.1f}e-18  # 阿秒")

print(f"\n选项C：看半周期（颜色反转）")
print(f"  t_snapshot = {T/2*1e15:.6f}e-15  # 飞秒")

print(f"\n选项D：动态观察（制作动画）")
print(f"  for t in np.linspace(0, T, 50):")
print(f"      # 计算每个时刻的场分布")
print(f"      # 保存为动画帧")

print("\n" + "="*60)
print("快速参考卡")
print("="*60)
print(f"λ = {WAVELENGTH*1e9:.0f} nm:")
print(f"  T = {T*1e15:.3f} fs = {T*1e18:.0f} as")
print(f"  T/4 = {T/4*1e15:.3f} fs = {T/4*1e18:.0f} as")
print(f"  T/2 = {T/2*1e15:.3f} fs = {T/2*1e18:.0f} as")
print("\n代码中使用：")
print(f"  t_snapshot = 0           # 初始")
print(f"  t_snapshot = {T/4:.2e}  # T/4")
print(f"  t_snapshot = {T/2:.2e}  # T/2")
print("="*60)

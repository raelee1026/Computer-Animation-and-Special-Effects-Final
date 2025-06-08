import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
from collections import deque

class RopeSimulationAnalysis:
    def __init__(self, num_particles=30, rest_length=0.02, dt=0.01, mass=0.01):
        self.num_particles = num_particles
        self.rest_length = rest_length
        self.dt = dt
        self.mass = mass
        self.gravity = np.array([0, -10.0])
        
        # 性能分析相關變數
        self.metrics = {
            'ftl': {'computation_time': deque(maxlen=100), 'energy': deque(maxlen=100), 
                   'constraint_error': deque(maxlen=100), 'stability': deque(maxlen=100)},
            'pbd': {'computation_time': deque(maxlen=100), 'energy': deque(maxlen=100), 
                   'constraint_error': deque(maxlen=100), 'stability': deque(maxlen=100)},
            'euler': {'computation_time': deque(maxlen=100), 'energy': deque(maxlen=100), 
                     'constraint_error': deque(maxlen=100), 'stability': deque(maxlen=100)}
        }
        
        self.reset_ropes()
        
        # Mouse interaction
        self.is_dragging = False
        self.drag_pos = np.array([0.0, 0.6])
        self.top_particle_pos = np.array([0.0, 0.6])
        
    def reset_ropes(self):
        """初始化三種方法的位置和速度"""
        self.top_particle_pos = np.array([0.0, 0.6])
        
        # FTL system (藍色)
        self.x_ftl = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_ftl[i] = self.top_particle_pos + np.array([-0.05 - i * 0.005, -i * self.rest_length])
        self.v_ftl = np.zeros((self.num_particles, 2))
        
        # PBD system (綠色)
        self.x_pbd = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_pbd[i] = self.top_particle_pos + np.array([0.0, -i * self.rest_length])
        self.v_pbd = np.zeros((self.num_particles, 2))
        
        # Symplectic Euler system (紅色)
        self.x_euler = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_euler[i] = self.top_particle_pos + np.array([0.05 + i * 0.005, -i * self.rest_length])
        self.v_euler = np.zeros((self.num_particles, 2))

    def calculate_energy(self, x, v):
        """計算總能量（動能+重力勢能）"""
        kinetic = 0.5 * self.mass * np.sum(v**2)
        potential = 0
        for i in range(self.num_particles):
            potential += self.mass * abs(self.gravity[1]) * x[i, 1]
        return kinetic + potential

    def calculate_constraint_error(self, x):
        """計算約束誤差（每段長度與理想長度的差異）"""
        errors = []
        for i in range(1, self.num_particles):
            segment_length = np.linalg.norm(x[i] - x[i-1])
            error = abs(segment_length - self.rest_length)
            errors.append(error)
        return np.mean(errors)

    def calculate_stability(self, v):
        """計算穩定性（速度變化的標準差）"""
        velocity_magnitudes = np.linalg.norm(v, axis=1)
        return np.std(velocity_magnitudes)

    def simulate_ftl(self):
        """Follow-The-Leader (FTL) 方法"""
        start_time = time.perf_counter()
        
        x, v = self.x_ftl, self.v_ftl
        dt = self.dt
        l0 = self.rest_length
        s_damping = 0.8
        gravity = self.gravity
        mass = self.mass

        x[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)

        f = np.zeros_like(x)
        for i in range(1, self.num_particles):
            f[i] = gravity * mass
        p = x + dt * v + (dt ** 2) * f / mass

        d = np.zeros_like(x)
        p[0] = x[0].copy()
        for i in range(1, self.num_particles):
            dir_vec = p[i] - p[i - 1]
            current_len = np.linalg.norm(dir_vec) + 1e-8
            correction = (l0 - current_len) * (dir_vec / current_len)
            d[i] = correction
            p[i] += correction

        for i in range(1, self.num_particles - 1):
            v[i] = (p[i] - x[i]) / dt + s_damping * (-d[i + 1]) / dt
        v[-1] = (p[-1] - x[-1]) / dt

        x[:] = p.copy()
        
        # 記錄性能指標
        computation_time = time.perf_counter() - start_time
        energy = self.calculate_energy(x, v)
        constraint_error = self.calculate_constraint_error(x)
        stability = self.calculate_stability(v)
        
        self.metrics['ftl']['computation_time'].append(computation_time)
        self.metrics['ftl']['energy'].append(energy)
        self.metrics['ftl']['constraint_error'].append(constraint_error)
        self.metrics['ftl']['stability'].append(stability)

    def simulate_pbd(self):
        """Position Based Dynamics (PBD) 方法"""
        start_time = time.perf_counter()
        
        x, v = self.x_pbd, self.v_pbd

        for i in range(1, self.num_particles):
            v[i] += self.dt * self.gravity

        p = x + self.dt * v
        p[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)

        for iteration in range(25):
            for i in range(1, self.num_particles):
                delta = p[i] - p[i - 1]
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    continue
                direction = delta / dist
                correction = (dist - self.rest_length) * direction

                if i > 1:
                    p[i - 1] += 0.5 * correction
                p[i] -= 0.5 * correction

        for i in range(1, self.num_particles):
            v[i] = (p[i] - x[i]) / self.dt
            v[i] *= 0.95

        x[:] = p.copy()
        
        # 記錄性能指標
        computation_time = time.perf_counter() - start_time
        energy = self.calculate_energy(x, v)
        constraint_error = self.calculate_constraint_error(x)
        stability = self.calculate_stability(v)
        
        self.metrics['pbd']['computation_time'].append(computation_time)
        self.metrics['pbd']['energy'].append(energy)
        self.metrics['pbd']['constraint_error'].append(constraint_error)
        self.metrics['pbd']['stability'].append(stability)

    def simulate_symplectic_euler(self):
        """Symplectic Euler 方法"""
        start_time = time.perf_counter()
        
        x, v = self.x_euler, self.v_euler
        x[0] = self.top_particle_pos.copy()
        
        stiff = 300
        substeps = 40
        sub_dt = self.dt / substeps
        
        for substep in range(substeps):
            forces = np.zeros_like(x)
            
            for i in range(1, self.num_particles):
                forces[i] += self.mass * self.gravity
            
            for i in range(1, self.num_particles):
                spring_vec = x[i] - x[i-1]
                spring_length = np.linalg.norm(spring_vec)
                
                if spring_length > 1e-8:
                    spring_dir = spring_vec / spring_length
                    spring_force = stiff * (spring_length - self.rest_length) * spring_dir
                    
                    if i > 1:
                        forces[i-1] += spring_force
                    forces[i] -= spring_force
            
            for i in range(1, self.num_particles):
                v[i] += sub_dt * forces[i] / self.mass
                v[i] *= 0.9995
            
            for i in range(1, self.num_particles):
                x[i] += sub_dt * v[i]
        
        x[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)
        
        # 記錄性能指標
        computation_time = time.perf_counter() - start_time
        energy = self.calculate_energy(x, v)
        constraint_error = self.calculate_constraint_error(x)
        stability = self.calculate_stability(v)
        
        self.metrics['euler']['computation_time'].append(computation_time)
        self.metrics['euler']['energy'].append(energy)
        self.metrics['euler']['constraint_error'].append(constraint_error)
        self.metrics['euler']['stability'].append(stability)

    def get_performance_summary(self):
        """獲取性能摘要統計"""
        summary = {}
        for method in ['ftl', 'pbd', 'euler']:
            summary[method] = {}
            for metric in ['computation_time', 'energy', 'constraint_error', 'stability']:
                if len(self.metrics[method][metric]) > 0:
                    data = list(self.metrics[method][metric])
                    summary[method][metric] = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'min': np.min(data),
                        'max': np.max(data)
                    }
        return summary

# 創建模擬實例
sim = RopeSimulationAnalysis()

# 設置可視化 - 2x2 子圖佈局
fig = plt.figure(figsize=(16, 12))

# 主模擬視圖
ax_main = plt.subplot(2, 3, (1, 4))
line_ftl, = ax_main.plot([], [], 'o-', color='blue', label='FTL (Dynamic)', 
                         markersize=4, linewidth=2, alpha=0.8)
line_pbd, = ax_main.plot([], [], 'o-', color='green', label='PBD (Position Based)', 
                         markersize=4, linewidth=2, alpha=0.8)
line_euler, = ax_main.plot([], [], 'o-', color='red', label='Symplectic Euler (Spring)', 
                          markersize=4, linewidth=2, alpha=0.8)
top_marker, = ax_main.plot([], [], 'ko', markersize=8, label='Shared Top Particle')

ax_main.set_xlim(-0.8, 0.8)
ax_main.set_ylim(-0.5, 0.8)
ax_main.set_aspect('equal')
ax_main.set_title('Rope Simulation: Three Methods Comparison\n(Click and drag to move top particle)', fontsize=14)
ax_main.legend(loc='upper right')
ax_main.grid(True, alpha=0.3)
ax_main.set_xlabel('X Position (m)')
ax_main.set_ylabel('Y Position (m)')

# Computation time comparison
ax_time = plt.subplot(2, 3, 2)
ax_time.set_title('Computation Time Comparison')
ax_time.set_ylabel('Time (seconds)')
ax_time.set_xlabel('Frame')

# Constraint error comparison
ax_error = plt.subplot(2, 3, 3)
ax_error.set_title('Constraint Error Comparison')
ax_error.set_ylabel('Error (m)')
ax_error.set_xlabel('Frame')

# Energy comparison
ax_energy = plt.subplot(2, 3, 5)
ax_energy.set_title('Total Energy Comparison')
ax_energy.set_ylabel('Energy (J)')
ax_energy.set_xlabel('Frame')

# Stability comparison
ax_stability = plt.subplot(2, 3, 6)
ax_stability.set_title('Stability Comparison')
ax_stability.set_ylabel('Velocity Standard Deviation')
ax_stability.set_xlabel('Frame')

plt.tight_layout()

# 鼠標事件處理
def on_press(event):
    if event.inaxes == ax_main:
        sim.is_dragging = True

def on_release(event):
    sim.is_dragging = False

def on_motion(event):
    if sim.is_dragging and event.inaxes == ax_main and event.xdata is not None:
        sim.drag_pos = np.array([event.xdata, event.ydata])
        sim.top_particle_pos = np.array([event.xdata, event.ydata])

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# 動畫更新函數
frame_count = 0
def update(frame):
    global frame_count
    frame_count += 1
    
    # 運行所有模擬
    sim.simulate_ftl()
    sim.simulate_pbd()
    sim.simulate_symplectic_euler()
    
    # 更新主視圖
    line_ftl.set_data(sim.x_ftl[:, 0], sim.x_ftl[:, 1])
    line_pbd.set_data(sim.x_pbd[:, 0], sim.x_pbd[:, 1])
    line_euler.set_data(sim.x_euler[:, 0], sim.x_euler[:, 1])
    top_marker.set_data([sim.top_particle_pos[0]], [sim.top_particle_pos[1]])
    
    # 更新性能圖表（每10幀更新一次以提高性能）
    if frame_count % 10 == 0:
        # 清除舊的圖表
        ax_time.clear()
        ax_error.clear()
        ax_energy.clear()
        ax_stability.clear()
        
        # Computation time
        if len(sim.metrics['ftl']['computation_time']) > 0:
            frames = range(len(sim.metrics['ftl']['computation_time']))
            ax_time.plot(frames, list(sim.metrics['ftl']['computation_time']), 'b-', label='FTL')
            ax_time.plot(frames, list(sim.metrics['pbd']['computation_time']), 'g-', label='PBD')
            ax_time.plot(frames, list(sim.metrics['euler']['computation_time']), 'r-', label='Euler')
            ax_time.set_title('Computation Time Comparison')
            ax_time.set_ylabel('Time (seconds)')
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
        
        # Constraint error
        if len(sim.metrics['ftl']['constraint_error']) > 0:
            frames = range(len(sim.metrics['ftl']['constraint_error']))
            ax_error.plot(frames, list(sim.metrics['ftl']['constraint_error']), 'b-', label='FTL')
            ax_error.plot(frames, list(sim.metrics['pbd']['constraint_error']), 'g-', label='PBD')
            ax_error.plot(frames, list(sim.metrics['euler']['constraint_error']), 'r-', label='Euler')
            ax_error.set_title('Constraint Error Comparison')
            ax_error.set_ylabel('Error (m)')
            ax_error.legend()
            ax_error.grid(True, alpha=0.3)
        
        # Energy
        if len(sim.metrics['ftl']['energy']) > 0:
            frames = range(len(sim.metrics['ftl']['energy']))
            ax_energy.plot(frames, list(sim.metrics['ftl']['energy']), 'b-', label='FTL')
            ax_energy.plot(frames, list(sim.metrics['pbd']['energy']), 'g-', label='PBD')
            ax_energy.plot(frames, list(sim.metrics['euler']['energy']), 'r-', label='Euler')
            ax_energy.set_title('Total Energy Comparison')
            ax_energy.set_ylabel('Energy (J)')
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)
        
        # Stability
        if len(sim.metrics['ftl']['stability']) > 0:
            frames = range(len(sim.metrics['ftl']['stability']))
            ax_stability.plot(frames, list(sim.metrics['ftl']['stability']), 'b-', label='FTL')
            ax_stability.plot(frames, list(sim.metrics['pbd']['stability']), 'g-', label='PBD')
            ax_stability.plot(frames, list(sim.metrics['euler']['stability']), 'r-', label='Euler')
            ax_stability.set_title('Stability Comparison')
            ax_stability.set_ylabel('Velocity Standard Deviation')
            ax_stability.legend()
            ax_stability.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    return line_ftl, line_pbd, line_euler, top_marker

# Instructions
print("Rope Simulation - Three Methods Performance Comparison")
print("="*55)
print("Click and drag anywhere to move the shared top particle")
print("\nMethod descriptions:")
print("• Blue (FTL): Fast Tangent Loop - dynamic constraints")
print("• Green (PBD): Position Based Dynamics - constraint projection") 
print("• Red (Symplectic Euler): Spring system with high stiffness")
print("\nPerformance metrics:")
print("• Computation Time: Time cost per frame")
print("• Constraint Error: Deviation of rope segment length from ideal length")
print("• Total Energy: Kinetic energy + Gravitational potential energy")
print("• Stability: Standard deviation of velocity changes")

# 創建並運行動畫
ani = animation.FuncAnimation(fig, update, frames=1000, interval=50, blit=False)

# 運行一段時間後顯示性能摘要表格
def show_performance_table():
    """Display performance analysis table"""
    # Wait for some data accumulation
    time.sleep(10)  # Wait 10 seconds
    
    summary = sim.get_performance_summary()
    
    # Create DataFrame for table display
    methods = ['FTL', 'PBD', 'Symplectic Euler']
    method_keys = ['ftl', 'pbd', 'euler']
    
    table_data = []
    for i, method_key in enumerate(method_keys):
        if method_key in summary:
            row = {
                'Method': methods[i],
                'Avg Computation Time (ms)': f"{summary[method_key]['computation_time']['mean']*1000:.3f}",
                'Computation Time Std (ms)': f"{summary[method_key]['computation_time']['std']*1000:.3f}",
                'Avg Constraint Error (mm)': f"{summary[method_key]['constraint_error']['mean']*1000:.3f}",
                'Constraint Error Std (mm)': f"{summary[method_key]['constraint_error']['std']*1000:.3f}",
                'Avg Energy (J)': f"{summary[method_key]['energy']['mean']:.3f}",
                'Stability Index': f"{summary[method_key]['stability']['mean']:.3f}"
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    print("\n" + "="*120)
    print("Performance Analysis Summary Table")
    print("="*120)
    print(df.to_string(index=False))
    print("\nPerformance analysis notes:")
    print("• Lower computation time is better (higher efficiency)")
    print("• Lower constraint error is better (higher accuracy)")
    print("• Energy should be relatively stable (physical realism)")
    print("• Lower stability index is better (numerical stability)")

# Run performance table display in background thread
import threading
table_thread = threading.Thread(target=show_performance_table)
table_thread.daemon = True
table_thread.start()

plt.show()
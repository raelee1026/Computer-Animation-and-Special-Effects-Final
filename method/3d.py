import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
num_strands = 100
num_particles = 40   # 更滑順
l0 = 0.5
head_radius = 5.0
dt = 0.02
s_damping = 0.9
s_friction = 0.1
s_repulsion = 0.2
g = np.array([0, -9.81, 0])

# --- Initialize strands ---
x = np.zeros((num_strands, num_particles, 3))
v = np.zeros((num_strands, num_particles, 3))
f = np.zeros((num_strands, num_particles, 3))

# Place base points on head sphere
for s in range(num_strands):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi / 2)  # upper hemisphere
    base_pos = np.array([
        head_radius * np.sin(phi) * np.cos(theta),
        head_radius * np.cos(phi),
        head_radius * np.sin(phi) * np.sin(theta)
    ])
    normal = base_pos / np.linalg.norm(base_pos)

    for i in range(num_particles):
        x[s, i] = base_pos + normal * l0 * i

# --- Background grid for hair-hair interaction ---
grid_size = 30
grid_res = 1.0
density_field = np.zeros((grid_size, grid_size, grid_size))
velocity_field = np.zeros((grid_size, grid_size, grid_size, 3))

def get_grid_index(pos):
    i = np.clip(int(pos[0] / grid_res) + grid_size // 2, 0, grid_size - 1)
    j = np.clip(int(pos[1] / grid_res) + grid_size // 2, 0, grid_size - 1)
    k = np.clip(int(pos[2] / grid_res) + grid_size // 2, 0, grid_size - 1)
    return i, j, k

# --- Setup 3D visualization ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], 'o-', lw=1)[0] for _ in range(num_strands)]

# Generate head sphere mesh
u = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 50)
head_x = head_radius * np.outer(np.cos(u), np.sin(v_sphere))
head_y = head_radius * np.outer(np.ones(np.size(u)), np.cos(v_sphere))
head_z = head_radius * np.outer(np.sin(u), np.sin(v_sphere))

# --- Simulation step ---
def simulate_step(frame):
    global x, v

    # Wind force: dynamic
    wind_strength = 20.0
    wind_dir = np.array([
        np.sin(0.5 * frame * dt),
        0.0,
        np.cos(0.3 * frame * dt)
    ])
    wind_force = wind_strength * wind_dir

    # --- For each strand ---
    for s in range(num_strands):
        for i in range(num_particles):
            f[s, i] = g + wind_force

        # --- Predict position ---
        p = x[s] + dt * v[s] + (dt ** 2) * f[s]

        # --- FTL projection ---
        d = np.zeros((num_particles, 3))
        base_pos = x[s, 0]
        base_dir = base_pos / np.linalg.norm(base_pos)
        p[0] = base_dir * head_radius

        for i in range(1, num_particles):
            dir_vec = p[i] - p[i - 1]
            current_len = np.linalg.norm(dir_vec) + 1e-8
            correction = (l0 - current_len) * (dir_vec / current_len)
            d[i] = correction
            p[i] += correction

        # --- Velocity correction (Eq. 9) ---
        for i in range(1, num_particles - 1):
            v[s, i] = (p[i] - x[s, i]) / dt + s_damping * ( -d[i + 1] ) / dt
        v[s, -1] = (p[-1] - x[s, -1]) / dt

        x[s] = p.copy()

    # --- Hair-hair interaction ---
    density_field.fill(0)
    velocity_field.fill(0)

    for s in range(num_strands):
        for i in range(num_particles):
            gi, gj, gk = get_grid_index(x[s, i])
            density_field[gi, gj, gk] += 1
            velocity_field[gi, gj, gk] += v[s, i]

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if density_field[i, j, k] > 0:
                    velocity_field[i, j, k] /= density_field[i, j, k]

    for s in range(num_strands):
        for i in range(1, num_particles):
            gi, gj, gk = get_grid_index(x[s, i])

            v_grid = velocity_field[gi, gj, gk]
            v[s, i] = (1 - s_friction) * v[s, i] + s_friction * v_grid

            grad_rho = np.zeros(3)
            if gi < grid_size - 1:
                grad_rho[0] += density_field[gi + 1, gj, gk] - density_field[gi, gj, gk]
            if gj < grid_size - 1:
                grad_rho[1] += density_field[gi, gj + 1, gk] - density_field[gi, gj, gk]
            if gk < grid_size - 1:
                grad_rho[2] += density_field[gi, gj, gk + 1] - density_field[gi, gj, gk]

            norm_grad = np.linalg.norm(grad_rho) + 1e-8
            g_vec = grad_rho / norm_grad

            v[s, i] += (s_repulsion * g_vec) / dt

    # --- Head collision ---
    for s in range(num_strands):
        for i in range(1, num_particles):
            pos = x[s, i]
            dist = np.linalg.norm(pos)
            if dist < head_radius:
                x[s, i] = (pos / dist) * head_radius

# --- Animation update ---
def update(frame):
    simulate_step(frame)

    ax.clear()
    ax.plot_surface(head_x, head_z, head_y, color='lightgray', alpha=0.3, edgecolor='none')

    for s in range(num_strands):
        ax.plot(x[s, :, 0], x[s, :, 2], x[s, :, 1], 'o-', lw=1, color='black')

    margin = 5
    ax.set_xlim(-head_radius - margin, head_radius + margin)
    ax.set_ylim(-head_radius - margin, head_radius + margin)
    ax.set_zlim(-head_radius - margin, head_radius + num_particles * l0 + margin)

    ax.set_title('3D FTL Hair on Head - Frame {}'.format(frame))

    return lines

# --- Create animation ---
ani = animation.FuncAnimation(fig, update, frames=1000, interval=30, blit=False)

plt.show()

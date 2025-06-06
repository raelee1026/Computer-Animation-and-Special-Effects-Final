import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
num_strands = 20            # Number of hair strands
num_particles = 15          # Number of particles per strand
l0 = 0.8                    # Rest length between particles
dt = 0.05                   # Time step
s_damping = 0.9             # DFTL velocity correction damping
s_friction = 0.1            # Hair-hair friction parameter
s_repulsion = 0.3           # Hair-hair repulsion parameter
gravity = np.array([0, -9.81])  # Gravity force

# --- Initialize hair strands ---
x = np.zeros((num_strands, num_particles, 2))  # Positions
v = np.zeros((num_strands, num_particles, 2))  # Velocities
f = np.zeros((num_strands, num_particles, 2))  # External forces

# Arrange initial hair strand positions horizontally
spacing = 1.0
for s in range(num_strands):
    base_x = (s - num_strands / 2) * spacing
    for i in range(num_particles):
        x[s, i] = np.array([base_x, l0 * (num_particles - i)])

# --- Background grid for hair-hair interaction ---
grid_size = 30
grid_resolution = 1.0
density_field = np.zeros((grid_size, grid_size))
velocity_field = np.zeros((grid_size, grid_size, 2))

def get_grid_index(pos):
    i = np.clip(int(pos[0] / grid_resolution) + grid_size // 2, 0, grid_size - 1)
    j = np.clip(int(pos[1] / grid_resolution), 0, grid_size - 1)
    return i, j

# --- Setup visualization ---
fig, ax = plt.subplots(figsize=(8, 10))
lines = []
for _ in range(num_strands):
    line, = ax.plot([], [], 'o-', lw=2)
    lines.append(line)

ax.set_xlim(-num_strands * spacing, num_strands * spacing)
ax.set_ylim(-5, num_particles * l0 + 5)
ax.set_aspect('equal')
ax.set_title('Dynamic FTL Hair/Fur Simulation (2D Animation - Full Scene)')

# --- Simulation step function ---
def simulate_step():
    global x, v

    # 1️⃣ Apply gravity
    for s in range(num_strands):
        for i in range(num_particles):
            f[s, i] = gravity

    # 2️⃣ Semi-implicit Euler: predict position
    p = x + dt * v + (dt ** 2) * f

    # 3️⃣ FTL constraint projection (Section 3.1)
    d = np.zeros((num_strands, num_particles, 2))  # Correction vectors
    for s in range(num_strands):
        p[s, 0] = x[s, 0]  # First particle fixed
        for i in range(1, num_particles):
            dir_vec = p[s, i] - p[s, i - 1]
            current_len = np.linalg.norm(dir_vec) + 1e-8
            correction = (l0 - current_len) * (dir_vec / current_len)
            d[s, i] = correction
            p[s, i] += correction

    # 4️⃣ Velocity correction (Section 3.3, Eq. 9)
    for s in range(num_strands):
        for i in range(1, num_particles - 1):
            v[s, i] = (p[s, i] - x[s, i]) / dt + s_damping * ( -d[s, i + 1] ) / dt
        v[s, -1] = (p[s, -1] - x[s, -1]) / dt

    # 5️⃣ Hair-hair interaction (Section 3.5)
    density_field.fill(0)
    velocity_field.fill(0)

    # Scatter all particles into grid
    for s in range(num_strands):
        for i in range(num_particles):
            gi, gj = get_grid_index(p[s, i])
            density_field[gi, gj] += 1
            velocity_field[gi, gj] += v[s, i]

    # Normalize grid velocity
    for i in range(grid_size):
        for j in range(grid_size):
            if density_field[i, j] > 0:
                velocity_field[i, j] /= density_field[i, j]

    # Apply friction + repulsion
    for s in range(num_strands):
        for i in range(1, num_particles):
            gi, gj = get_grid_index(p[s, i])

            # Friction (Eq. 10)
            v_grid = velocity_field[gi, gj]
            v[s, i] = (1 - s_friction) * v[s, i] + s_friction * v_grid

            # Repulsion (Eq. 11)
            grad_rho = np.array([0.0, 0.0])
            if gi < grid_size - 1:
                grad_rho[0] += density_field[gi + 1, gj] - density_field[gi, gj]
            if gj < grid_size - 1:
                grad_rho[1] += density_field[gi, gj + 1] - density_field[gi, gj]

            norm_grad = np.linalg.norm(grad_rho) + 1e-8
            g = grad_rho / norm_grad

            v[s, i] += (s_repulsion * g) / dt

    # 6️⃣ Update positions
    x[:] = p.copy()

# --- Animation update function ---
def update(frame):
    simulate_step()

    # Update line data for each strand
    for s in range(num_strands):
        lines[s].set_data(x[s, :, 0], x[s, :, 1])
    return lines

# --- Create animation ---
ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)

plt.show()

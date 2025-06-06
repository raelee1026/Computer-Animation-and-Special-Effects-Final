import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
num_particles = 20          # Number of particles per strand
l0 = 1.0                    # Rest length between particles
dt = 0.05                   # Time step
s_damping = 0.9             # DFTL velocity correction damping
s_friction = 0.1            # Hair-hair friction parameter
s_repulsion = 0.2           # Hair-hair repulsion parameter
gravity = np.array([0, -9.81])  # Gravity force

# --- Initialize hair strand ---
x = np.zeros((num_particles, 2))  # Positions
v = np.zeros((num_particles, 2))  # Velocities
f = np.zeros((num_particles, 2))  # External forces

# Initial position: vertical line with spacing l0
for i in range(num_particles):
    x[i] = np.array([0.0, l0 * (num_particles - i)])

# --- Background grid for hair-hair interaction ---
grid_size = 10
grid_resolution = 1.0
density_field = np.zeros((grid_size, grid_size))
velocity_field = np.zeros((grid_size, grid_size, 2))

def get_grid_index(pos):
    i = np.clip(int(pos[0] / grid_resolution), 0, grid_size - 1)
    j = np.clip(int(pos[1] / grid_resolution), 0, grid_size - 1)
    return i, j

# --- Simulation loop ---
num_steps = 200

# For visualization
history = []

for step in range(num_steps):
    # 1️⃣ Apply gravity
    for i in range(num_particles):
        f[i] = gravity

    # 2️⃣ Semi-implicit Euler: predict position
    p = x + dt * v + (dt ** 2) * f

    # 3️⃣ FTL constraint projection
    d = np.zeros((num_particles, 2))  # Correction vectors

    # First particle fixed
    p[0] = x[0]

    for i in range(1, num_particles):
        dir_vec = p[i] - p[i - 1]
        current_len = np.linalg.norm(dir_vec) + 1e-8
        correction = (l0 - current_len) * (dir_vec / current_len)
        d[i] = correction
        p[i] += correction

    # 4️⃣ Velocity correction (Eq. 9)
    for i in range(1, num_particles - 1):
        v[i] = (p[i] - x[i]) / dt + s_damping * ( -d[i + 1] ) / dt

    # Last particle (no d[i+1])
    v[-1] = (p[-1] - x[-1]) / dt

    # 5️⃣ Hair-hair interaction (simple density field)
    density_field.fill(0)
    velocity_field.fill(0)

    # Scatter particles into grid
    for i in range(num_particles):
        gi, gj = get_grid_index(p[i])
        density_field[gi, gj] += 1
        velocity_field[gi, gj] += v[i]

    # Normalize grid velocity
    for i in range(grid_size):
        for j in range(grid_size):
            if density_field[i, j] > 0:
                velocity_field[i, j] /= density_field[i, j]

    # Apply friction + repulsion
    for i in range(1, num_particles):
        gi, gj = get_grid_index(p[i])

        # Friction (Eq. 10)
        v_grid = velocity_field[gi, gj]
        v[i] = (1 - s_friction) * v[i] + s_friction * v_grid

        # Repulsion (Eq. 11) -- approximate gradient as neighbor diff
        grad_rho = np.array([0.0, 0.0])
        if gi < grid_size - 1:
            grad_rho[0] += density_field[gi + 1, gj] - density_field[gi, gj]
        if gj < grid_size - 1:
            grad_rho[1] += density_field[gi, gj + 1] - density_field[gi, gj]

        norm_grad = np.linalg.norm(grad_rho) + 1e-8
        g = grad_rho / norm_grad

        v[i] += (s_repulsion * g) / dt

    # 6️⃣ Update positions
    x = p.copy()

    # Save history
    history.append(x.copy())

# --- Visualization ---

fig, ax = plt.subplots(figsize=(6, 8))

for t in range(0, len(history), 10):
    pos = history[t]
    ax.plot(pos[:, 0], pos[:, 1], marker='o', label=f'Step {t}')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 25)
ax.set_aspect('equal')
ax.set_title('Dynamic FTL Hair Simulation (with velocity correction and hair-hair interaction)')
ax.legend()
plt.show()

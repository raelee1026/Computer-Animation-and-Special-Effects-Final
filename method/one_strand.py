import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
num_particles = 20
l0 = 1.0
dt = 0.05
s_damping = 0.9
s_friction = 0.0  # 不用 hair-hair interaction，單條 strand
s_repulsion = 0.0
gravity = np.array([0, -9.81])

# --- Initialize strand ---
x = np.zeros((num_particles, 2))
v = np.zeros((num_particles, 2))
f = np.zeros((num_particles, 2))

# Initial vertical position
for i in range(num_particles):
    x[i] = np.array([0.0, l0 * (num_particles - i)])

# --- Setup visualization ---
fig, ax = plt.subplots(figsize=(6, 8))
line, = ax.plot([], [], 'o-', lw=2)

ax.set_xlim(-10, 10)
ax.set_ylim(-5, num_particles * l0 + 5)
ax.set_aspect('equal')
ax.set_title('Dynamic FTL - One Hair Strand Large Swing')

# --- Simulation step ---
def simulate_step(frame):
    global x, v

    # 1️⃣ Top particle external motion - large left-right swing
    swing_amplitude = 5.0
    swing_speed = 1.0
    top_x = swing_amplitude * np.sin(swing_speed * frame * dt)
    x[0] = np.array([top_x, l0 * num_particles])  # fixed top position
    v[0] = np.array([0, 0])  # velocity 0 because we fix x[0]

    # 2️⃣ Apply gravity
    for i in range(1, num_particles):
        f[i] = gravity

    # 3️⃣ Predict position
    p = x + dt * v + (dt ** 2) * f

    # 4️⃣ FTL constraint projection
    d = np.zeros((num_particles, 2))
    p[0] = x[0]  # fixed

    for i in range(1, num_particles):
        dir_vec = p[i] - p[i - 1]
        current_len = np.linalg.norm(dir_vec) + 1e-8
        correction = (l0 - current_len) * (dir_vec / current_len)
        d[i] = correction
        p[i] += correction

    # 5️⃣ Velocity correction
    for i in range(1, num_particles - 1):
        v[i] = (p[i] - x[i]) / dt + s_damping * ( -d[i + 1] ) / dt
    v[-1] = (p[-1] - x[-1]) / dt

    # 6️⃣ Update positions
    x[:] = p.copy()

# --- Animation update ---
def update(frame):
    simulate_step(frame)
    line.set_data(x[:, 0], x[:, 1])
    return line,

# --- Create animation ---
ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)

plt.show()

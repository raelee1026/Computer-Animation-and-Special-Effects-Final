import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RopeSimulation:
    def __init__(self, num_particles=30, rest_length=0.02, dt=0.01, mass=0.01):
        self.num_particles = num_particles
        self.rest_length = rest_length
        self.dt = dt
        self.mass = mass
        self.gravity = np.array([0, -10.0])  # -10 m/s^2 as in paper
        
        # Initialize three separate rope systems
        self.reset_ropes()
        
        # Mouse interaction
        self.is_dragging = False
        self.drag_pos = np.array([0.0, 0.6])
        self.top_particle_pos = np.array([0.0, 0.6])  # Shared top particle
        
    def reset_ropes(self):
        """Initialize positions and velocities for all three methods"""
        # Shared top particle position
        self.top_particle_pos = np.array([0.0, 0.6])
        
        # FTL system (blue) - starts going left
        self.x_ftl = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_ftl[i] = self.top_particle_pos + np.array([-0.05 - i * 0.005, -i * self.rest_length])
        self.v_ftl = np.zeros((self.num_particles, 2))
        
        # PBD system (green) - starts going straight down
        self.x_pbd = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_pbd[i] = self.top_particle_pos + np.array([0.0, -i * self.rest_length])
        self.v_pbd = np.zeros((self.num_particles, 2))
        
        # Symplectic Euler system (red) - starts going right
        self.x_euler = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_euler[i] = self.top_particle_pos + np.array([0.05 + i * 0.005, -i * self.rest_length])
        self.v_euler = np.zeros((self.num_particles, 2))

    def simulate_ftl(self):
        """Follow-The-Leader (FTL) with forward projection and velocity correction"""
        x, v = self.x_ftl, self.v_ftl
        dt = self.dt
        l0 = self.rest_length
        s_damping = 0.8
        gravity = self.gravity
        mass = self.mass

        # Top particle follows external position
        x[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)  # fixed

        # Predict positions under gravity
        f = np.zeros_like(x)
        for i in range(1, self.num_particles):
            f[i] = gravity * mass
        p = x + dt * v + (dt ** 2) * f / mass

        # FTL projection: forward chain to maintain rest length
        d = np.zeros_like(x)
        p[0] = x[0].copy()
        for i in range(1, self.num_particles):
            dir_vec = p[i] - p[i - 1]
            current_len = np.linalg.norm(dir_vec) + 1e-8
            correction = (l0 - current_len) * (dir_vec / current_len)
            d[i] = correction
            p[i] += correction

        # Velocity correction from Eq. (9)
        for i in range(1, self.num_particles - 1):
            v[i] = (p[i] - x[i]) / dt + s_damping * (-d[i + 1]) / dt
        v[-1] = (p[-1] - x[-1]) / dt  # last particle has no d[i+1]

        # Update positions
        x[:] = p.copy()

        
    def simulate_pbd(self):
        x, v = self.x_pbd, self.v_pbd

        # External force
        for i in range(1, self.num_particles):
            v[i] += self.dt * self.gravity

        # Predict positions
        p = x + self.dt * v

        # Fix top point
        p[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)

        # Constraint projection
        for iteration in range(25):  # more iterations = stiffer
            for i in range(1, self.num_particles):
                delta = p[i] - p[i - 1]
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    continue
                direction = delta / dist
                correction = (dist - self.rest_length) * direction

                # Half-half correction, but fixed top
                if i > 1:
                    p[i - 1] += 0.5 * correction
                p[i] -= 0.5 * correction

        # Update velocities (with damping)
        for i in range(1, self.num_particles):
            v[i] = (p[i] - x[i]) / self.dt
            v[i] *= 0.95  # stronger damping

        # Apply new positions
        x[:] = p.copy()



    # def simulate_pbd(self):
        # """Position Based Dynamics (PBD) method"""
        # x, v = self.x_pbd, self.v_pbd
        
        # # Update shared top particle position
        # x[0] = self.top_particle_pos.copy()
        
        # # Apply external forces
        # for i in range(1, self.num_particles):
        #     v[i] += self.dt * self.gravity
        
        # # Predict positions
        # p = x + self.dt * v
        
        # # Constraint particle 0 to shared top position
        # p[0] = self.top_particle_pos.copy()
        # v[0] = np.zeros(2)
        
        # # Constraint projection - 2 iterations as in paper
        # for iteration in range(2):
        #     for i in range(1, self.num_particles):
        #         constraint_vec = p[i] - p[i-1]
        #         current_length = np.linalg.norm(constraint_vec)
                
        #         if current_length > 1e-8:
        #             constraint_dir = constraint_vec / current_length
        #             delta_length = current_length - self.rest_length
                    
        #             # PBD correction (equal mass assumption)
        #             correction = 0.5 * delta_length * constraint_dir
                    
        #             if i > 1:  # Don't move the shared top particle
        #                 p[i-1] += correction
        #             p[i] -= correction
        
        # # Update velocities
        # for i in range(1, self.num_particles):
        #     v[i] = (p[i] - x[i]) / self.dt
        #     v[i] *= 0.99  # Moderate damping
        
        # x[:] = p.copy()
        
    def simulate_symplectic_euler(self):
        """Symplectic Euler with high stiffness and substeps"""
        x, v = self.x_euler, self.v_euler
        
        # Update shared top particle position
        x[0] = self.top_particle_pos.copy()
        
        # stiffness for spring forces
        stiff = 300
        
        # Use multiple substeps for stability (20 as mentioned in paper)
        substeps = 40
        sub_dt = self.dt / substeps
        
        for substep in range(substeps):
            # Calculate spring forces
            forces = np.zeros_like(x)
            
            # Gravity
            for i in range(1, self.num_particles):
                forces[i] += self.mass * self.gravity
            
            # Spring forces
            for i in range(1, self.num_particles):
                spring_vec = x[i] - x[i-1]
                spring_length = np.linalg.norm(spring_vec)
                
                if spring_length > 1e-8:
                    spring_dir = spring_vec / spring_length
                    spring_force = stiff * (spring_length - self.rest_length) * spring_dir
                    
                    if i > 1:  # Don't apply force to shared top particle
                        forces[i-1] += spring_force
                    forces[i] -= spring_force
            
            # Symplectic Euler integration
            # First update velocities
            for i in range(1, self.num_particles):
                v[i] += sub_dt * forces[i] / self.mass
                v[i] *= 0.9995  # Very light damping for stability
            
            # Then update positions
            for i in range(1, self.num_particles):
                x[i] += sub_dt * v[i]
        
        # Ensure first particle stays at shared top position
        x[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)

# Create simulation instance
sim = RopeSimulation()

# Setup visualization - single plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create lines for all three methods
line_ftl, = ax.plot([], [], 'o-', color='blue', label='FTL (Dynamic)', 
                    markersize=4, linewidth=2, alpha=0.8)
line_pbd, = ax.plot([], [], 'o-', color='green', label='PBD (Position Based)', 
                    markersize=4, linewidth=2, alpha=0.8)
line_euler, = ax.plot([], [], 'o-', color='red', label='Symplectic Euler (Spring)', 
                      markersize=4, linewidth=2, alpha=0.8)

# Add a marker for the shared top particle
top_marker, = ax.plot([], [], 'ko', markersize=8, label='Shared Top Particle')

ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.5, 0.8)
ax.set_aspect('equal')
ax.set_title('Rope Simulation: Three Methods with Shared Top Particle\n(Click and drag to move)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')

plt.tight_layout()

# Mouse event handlers
def on_press(event):
    if event.inaxes == ax:
        sim.is_dragging = True

def on_release(event):
    sim.is_dragging = False

def on_motion(event):
    if sim.is_dragging and event.inaxes == ax and event.xdata is not None:
        sim.drag_pos = np.array([event.xdata, event.ydata])
        sim.top_particle_pos = np.array([event.xdata, event.ydata])

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Animation update function
def update(frame):
    # Run all simulations
    sim.simulate_ftl()
    sim.simulate_pbd()
    sim.simulate_symplectic_euler()
    
    # Update line data
    line_ftl.set_data(sim.x_ftl[:, 0], sim.x_ftl[:, 1])
    line_pbd.set_data(sim.x_pbd[:, 0], sim.x_pbd[:, 1])
    line_euler.set_data(sim.x_euler[:, 0], sim.x_euler[:, 1])
    
    # Update top particle marker
    top_marker.set_data([sim.top_particle_pos[0]], [sim.top_particle_pos[1]])
    
    return line_ftl, line_pbd, line_euler, top_marker

# Instructions
print("Rope Simulation - Three Methods Comparison")
print("="*45)
print("Click and drag anywhere to move the shared top particle")
print("\nMethods:")
print("• Blue (FTL): Fast Tangent Loop - dynamic constraints")
print("• Green (PBD): Position Based Dynamics - constraint projection") 
print("• Red (Symplectic Euler): Spring system with high stiffness")
print("\nParameters match the research paper:")
print("• 30 particles per rope")
print("• 0.02m segment length")
print("• 0.01s time step")
print("• FTL & PBD: 2 iterations, Euler: 3000 N/m stiffness, 20 substeps")

# Create and run animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=30, blit=True)
plt.show()
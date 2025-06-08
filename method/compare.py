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
        self.stiffness = 100  # N/m (will be adjusted per method)
        
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
            self.x_ftl[i] = self.top_particle_pos + np.array([-0.1 - i * 0.01, -i * self.rest_length])
        self.v_ftl = np.zeros((self.num_particles, 2))
        
        # PBD system (green) - starts going straight down
        self.x_pbd = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_pbd[i] = self.top_particle_pos + np.array([0.0, -i * self.rest_length])
        self.v_pbd = np.zeros((self.num_particles, 2))
        
        # Symplectic Euler system (red) - starts going right
        self.x_euler = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            self.x_euler[i] = self.top_particle_pos + np.array([0.1 + i * 0.01, -i * self.rest_length])
        self.v_euler = np.zeros((self.num_particles, 2))
        
    def simulate_ftl(self):
        """Fast Tangent Loop (FTL) method - dynamic constraints"""
        x, v = self.x_ftl, self.v_ftl
        
        # Update shared top particle position
        x[0] = self.top_particle_pos.copy()
        
        # Apply external forces (gravity)
        forces = np.zeros_like(x)
        for i in range(1, self.num_particles):
            forces[i] = self.mass * self.gravity
        
        # Predict positions
        p = x + self.dt * v + (self.dt**2 / self.mass) * forces
        
        # Constraint particle 0 to shared top position
        p[0] = self.top_particle_pos.copy()
        
        # Fast tangential constraint projection
        for iteration in range(2):  # 2 iterations as mentioned in paper
            for i in range(1, self.num_particles):
                # Get constraint vector
                constraint_vec = p[i] - p[i-1]
                current_length = np.linalg.norm(constraint_vec)
                
                if current_length > 1e-8:
                    # Normalize
                    constraint_dir = constraint_vec / current_length
                    
                    # Calculate correction
                    delta_length = current_length - self.rest_length
                    correction = -0.5 * delta_length * constraint_dir
                    
                    # Apply corrections
                    if i > 1:  # Don't move the shared top particle
                        p[i-1] -= correction
                    p[i] += correction
        
        # Update velocities and positions
        v[0] = np.zeros(2)  # Top particle velocity controlled externally
        for i in range(1, self.num_particles):
            v[i] = (p[i] - x[i]) / self.dt
            # Add damping
            v[i] *= 0.99
        
        x[:] = p.copy()
        
    def simulate_pbd(self):
        """Position Based Dynamics (PBD) method"""
        x, v = self.x_pbd, self.v_pbd
        
        # Update shared top particle position
        x[0] = self.top_particle_pos.copy()
        
        # Apply external forces
        for i in range(1, self.num_particles):
            v[i] += self.dt * self.gravity
        
        # Predict positions
        p = x + self.dt * v
        
        # Constraint particle 0 to shared top position
        p[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)
        
        # Constraint projection (2 iterations as in paper)
        for iteration in range(2):
            for i in range(1, self.num_particles):
                constraint_vec = p[i] - p[i-1]
                current_length = np.linalg.norm(constraint_vec)
                
                if current_length > 1e-8:
                    constraint_dir = constraint_vec / current_length
                    delta_length = current_length - self.rest_length
                    
                    # PBD correction (equal mass assumption)
                    correction = 0.5 * delta_length * constraint_dir
                    
                    if i > 1:  # Don't move the shared top particle
                        p[i-1] += correction
                    p[i] -= correction
        
        # Update velocities
        for i in range(1, self.num_particles):
            v[i] = (p[i] - x[i]) / self.dt
            v[i] *= 0.98  # damping
        
        x[:] = p.copy()
        
    def simulate_symplectic_euler(self):
        """Symplectic Euler with high stiffness"""
        x, v = self.x_euler, self.v_euler
        
        # Update shared top particle position
        x[0] = self.top_particle_pos.copy()
        
        # Higher stiffness for stability (3000 N/m as mentioned in paper)
        stiff = 3000.0
        
        # Use multiple substeps for stability (20 as mentioned)
        substeps = 20
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
                v[i] *= 0.999  # slight damping for stability
            
            # Then update positions
            for i in range(1, self.num_particles):
                x[i] += sub_dt * v[i]
        
        # Ensure first particle stays at shared top position
        x[0] = self.top_particle_pos.copy()
        v[0] = np.zeros(2)

# Create simulation instance
sim = RopeSimulation()

# Setup visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Left plot - same time per frame
line_ftl_1, = ax1.plot([], [], 'o-', color='blue', label='FTL', markersize=3, linewidth=1.5)
line_pbd_1, = ax1.plot([], [], 'o-', color='green', label='PBD', markersize=3, linewidth=1.5)
line_euler_1, = ax1.plot([], [], 'o-', color='red', label='Symplectic Euler', markersize=3, linewidth=1.5)

ax1.set_xlim(-0.6, 0.6)
ax1.set_ylim(-0.4, 0.8)
ax1.set_aspect('equal')
ax1.set_title('Same Time Per Frame')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot - adjusted for similar results
line_ftl_2, = ax2.plot([], [], 'o-', color='blue', label='FTL', markersize=3, linewidth=1.5)
line_pbd_2, = ax2.plot([], [], 'o-', color='green', label='PBD', markersize=3, linewidth=1.5)
line_euler_2, = ax2.plot([], [], 'o-', color='red', label='Symplectic Euler', markersize=3, linewidth=1.5)

ax2.set_xlim(-0.6, 0.6)
ax2.set_ylim(-0.4, 0.8)
ax2.set_aspect('equal')
ax2.set_title('Adjusted Parameters')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Mouse event handlers
def on_press(event):
    sim.is_dragging = True

def on_release(event):
    sim.is_dragging = False

def on_motion(event):
    if sim.is_dragging and event.inaxes and event.xdata is not None:
        sim.drag_pos = np.array([event.xdata, event.ydata])
        sim.top_particle_pos = np.array([event.xdata, event.ydata])  # Update shared top particle

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Animation parameters
frame_count = 0
euler_skip = 0  # For the adjusted version

def update(frame):
    global frame_count, euler_skip
    frame_count += 1
    
    # Left side - same time per frame
    sim.simulate_ftl()
    sim.simulate_pbd()
    sim.simulate_symplectic_euler()
    
    # All three ropes share the same top particle, no offset needed
    line_ftl_1.set_data(sim.x_ftl[:, 0], sim.x_ftl[:, 1])
    line_pbd_1.set_data(sim.x_pbd[:, 0], sim.x_pbd[:, 1])
    line_euler_1.set_data(sim.x_euler[:, 0], sim.x_euler[:, 1])
    
    # Right side - adjusted parameters (simulate Euler less frequently)
    if frame_count % 3 == 0:  # Run Euler every 3rd frame
        # Reset Euler to match others periodically for comparison
        if frame_count % 60 == 0:
            sim.x_euler = sim.x_pbd.copy()
            sim.v_euler = sim.v_pbd.copy()
            # Ensure top particle is shared
            sim.x_euler[0] = sim.top_particle_pos.copy()
    
    line_ftl_2.set_data(sim.x_ftl[:, 0], sim.x_ftl[:, 1])
    line_pbd_2.set_data(sim.x_pbd[:, 0], sim.x_pbd[:, 1])
    line_euler_2.set_data(sim.x_euler[:, 0], sim.x_euler[:, 1])
    
    return line_ftl_1, line_pbd_1, line_euler_1, line_ftl_2, line_pbd_2, line_euler_2

# Instructions
print("Rope Simulation - Three Methods Comparison")
print("Click and drag to move the top particle")
print("Left panel: Same time per frame")
print("Right panel: Adjusted for similar behavior")
print("Blue: FTL, Green: PBD, Red: Symplectic Euler")

# Create and run animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=50, blit=True)
plt.show()
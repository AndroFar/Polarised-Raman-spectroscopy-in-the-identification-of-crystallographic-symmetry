import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches

#### Simulate coupled spring mass system
# assign constants
m1 = 5
m2 = 10
k1 = 20
k2 = 10
k3 = 20

# initial conditions
# resting mass position in respect to walls: m1 = -1, m2 = 1
x1_init = 1
x1_dot_init = 0
x2_init = 0
x2_dot_init = 0

# set simulation time and frames per second
t_final = 30
fps = 30

# system of differential equations
# y[0] = x1, y[1] = x1_dot, y[2] = x2, y[3] = x2_dot
def coupled_springs_ODE(t, y):
    x1, x1_dot, x2, x2_dot = y

    # equations derived in derive_CMSS.ipynb
    x1_ddot = (-k1*x1 + k2*(x2 - x1)) / m1
    x2_ddot = (-k2*(x2 - x1) - k3*x2) / m2

    return (x1_dot, x1_ddot, x2_dot, x2_ddot)

# solve ODE
sol = solve_ivp(coupled_springs_ODE, [0, t_final],
                (x1_init, x1_dot_init, x2_init, x2_dot_init),
                t_eval = np.linspace(0, t_final, t_final*fps+1))

# output
x1, x1_dot, x2, x2_dot = sol.y
t = sol.t

#### Animate coupled spring mass system
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('Coupled Spring-Mass System')
ax.set_xlabel('Position')
ax.grid(True, alpha=0.3)

# wall positions
wall1_x = -3
wall2_x = 3

# Equilibrium positions (where masses would be at rest)
eq1_x = wall1_x + 2  # Mass 1 equilibrium
eq2_x = wall2_x - 2  # Mass 2 equilibrium

# generate springs for both masses
def generate_spring_horizontal(n, length=1):
    """Generate a horizontal spring with n interior coils using an array"""
    data = np.zeros((2, n+2))   # [x; y]
    data[0, -1] = length    # end point
    data[1, :] = 0  # all y coordinates are 0

    for i in range(1, n+1):
        data[1, i] = 0.1 if i % 2 else -0.1 # coil vertical displacment
        data[0, i] = (i * length) / (n + 1) # coil spacing
    return data

# create springs
spring1_data = generate_spring_horizontal(20, length=1)
spring2_data = generate_spring_horizontal(20, length=1)
spring3_data = generate_spring_horizontal(20, length=1)

# create visualization objects
# walls
wall1 = ax.add_patch(plt.Rectangle((wall1_x-0.2, -0.8), 0.2, 1.6, fc='darkgray', ec='black', linewidth=2))
wall2 = ax.add_patch(plt.Rectangle((wall2_x, -0.8), 0.2, 1.6, fc='darkgray', ec='black', linewidth=2))

# springs
spring1 = Line2D(spring1_data[0, :], spring1_data[1, :], color='r', linewidth=2)
spring2 = Line2D(spring2_data[0, :], spring2_data[1, :], color='g', linewidth=2)
spring3 = Line2D(spring3_data[0, :], spring3_data[1, :], color='b', linewidth=2)

# masses
mass_size = 0.4
mass1 = ax.add_patch(patches.Rectangle((0, -mass_size/2), mass_size, mass_size,
                                       fc='blue', ec='black', zorder=3))
mass2 = ax.add_patch(patches.Rectangle((0, -mass_size/2), mass_size, mass_size,
                                       fc='red', ec='black', zorder=3))

# add equilibruim position markers
ax.axvline(x=eq1_x, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
ax.axvline(x=eq2_x, color='gray', linestyle='--', alpha=0.5)

# add all objects to plot
ax.add_line(spring1)
ax.add_line(spring2)
ax.add_line(spring3)

# add legend
custom_lines = [Line2D([0], [0], color='r', lw=2),
               Line2D([0], [0], color='g', lw=2),
               Line2D([0], [0], color='b', lw=2),
               Line2D([0], [0], color='blue', lw=0, marker='s', markersize=8),
               Line2D([0], [0], color='red', lw=0, marker='s', markersize=8)]
ax.legend(custom_lines, [f'(k1 = {k1})', f'(k2 = {k2})', f'(k3 = {k3})', f'(m1 = {m1})', f'(m2 = {m2})'], loc='upper right')

def animate(i):

    # Affine2D applies linear geometric transformations to plots. In this case the springs are stretched and compressed.

    # current positions - displacements from equilibrium
    pos1 = eq1_x + x1[i]  # Mass 1 position
    pos2 = eq2_x + x2[i]  # Mass 2 position
    
    # update mass positions
    mass1.set_xy((pos1 - mass_size/2, -mass_size/2))
    mass2.set_xy((pos2 - mass_size/2, -mass_size/2))
    
    # spring 1
    length1 = pos1 - wall1_x
    if length1 > 0:  # prevent negative lengths
        A1 = Affine2D().scale(length1, 1).get_matrix()
        spring1_data_new = np.matmul(A1, np.append(spring1_data, np.ones((1, 22)), axis=0))
        spring1.set_data(spring1_data_new[0, :] + wall1_x, spring1_data_new[1, :])
    
    # spring 2
    length2 = pos2 - pos1
    if length2 > 0:  # prevent negative lengths
        A2 = Affine2D().scale(length2, 1).get_matrix()
        spring2_data_new = np.matmul(A2, np.append(spring2_data, np.ones((1, 22)), axis=0))
        spring2.set_data(spring2_data_new[0, :] + pos1, spring2_data_new[1, :])
    
    # spring 3
    length3 = wall2_x - pos2
    if length3 > 0:  # prevent negative lengths
        A3 = Affine2D().scale(length3, 1).get_matrix()
        spring3_data_new = np.matmul(A3, np.append(spring3_data, np.ones((1, 22)), axis=0))
        spring3.set_data(spring3_data_new[0, :] + pos2, spring3_data_new[1, :])

    return spring1, spring2, spring3, mass1, mass2

# create animation
ani = animation.FuncAnimation(fig, animate, frames=len(t), blit=True, interval=1000/fps)
ffmpeg_writer = animation.FFMpegWriter(fps=fps)
ani.save('CMSS.mp4', writer=ffmpeg_writer)
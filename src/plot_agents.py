import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

def animate_agents(i, circles, positions):
    """Update the position of each circle."""
    for circle, position in zip(circles, positions[i]):
        circle.center = position
    return circles


def plot_movements(sim, anim_length=10, anim_fps=12):
    """Animate the movements of all agents."""
    skip_frames = max(int(len(sim.history)/(anim_length * anim_fps)), 1)

    # Extracting positions for plotting
    positions = np.array(sim.history[::skip_frames])  # Assuming sim.history is a list of positions for each step
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, sim.world_size)
    ax.set_ylim(0, sim.world_size)
    
    # Assuming the first set of positions to initialize the scatter plot
    colors = ['red' if agent.agent_type == 'A' else 'blue' if agent.agent_type == 'B' else 'green' for agent in sim.agents]
    circles = []
    for pos, color, agent in zip(positions[0], colors, sim.agents):
        circle = Circle(pos, radius=agent.radius, color=color)
        ax.add_patch(circle)
        circles.append(circle)
    
    anim = FuncAnimation(fig, animate_agents, frames=len(positions), interval=1e3/anim_fps, fargs=(circles, positions), blit=True)
    
    plt.show()
    return anim
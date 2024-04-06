import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

def animate_agents(i, agent_circles, positions):
    """Update the position of each agent's circles (both position and sensing radius)."""
    updated_artists = []
    for agent_data, position in zip(agent_circles, positions[i]):
        position_circle, sensing_circle = agent_data
        position_circle.center = position
        sensing_circle.center = position
        updated_artists.extend([position_circle, sensing_circle])
    return updated_artists

def plot_movements(sim, anim_length=10, anim_fps=12):
    """Animate the movements of all agents, including their sensing radius."""
    plot_sensing_radius = True

    skip_frames = max(int(len(sim.history)/(anim_length * anim_fps)), 1)
    positions = np.array(sim.history[::skip_frames])  # Assuming sim.history is a list of positions for each step
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, sim.world_size)
    ax.set_ylim(0, sim.world_size)

    colors = ['red' if agent.agent_type == 'A' else 'blue' if agent.agent_type == 'B' else 'green' for agent in sim.agents]
    agent_circles = []  # This will store tuples of (position_circle, sensing_circle) for each agent
    
    for pos, color, agent in zip(positions[0], colors, sim.agents):
        position_circle = Circle(pos, radius=agent.radius, color=color)
        ax.add_patch(position_circle)
        
        if plot_sensing_radius:
            sensing_circle = Circle(pos, radius=agent.sensing_radius, color=color, fill=False, linestyle='--')
            ax.add_patch(sensing_circle)
        else:
            sensing_circle = None
        
        agent_circles.append((position_circle, sensing_circle))
    
    anim = FuncAnimation(fig, animate_agents, frames=len(positions), interval=1e3/anim_fps, fargs=(agent_circles, positions), blit=True)
    
    ax.set_aspect('equal')
    plt.show()
    return anim

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch

def animate_agents(i, agent_circles, steps):
    """Update the position of each agent's circles (both position and sensing radius)."""
    updated_artists = []
    for agent_data, agent in zip(agent_circles, steps[i]):
        position_circle, sensing_circle = agent_data
        position_circle.center = agent.position
        sensing_circle.center = agent.position
        updated_artists.extend([position_circle, sensing_circle])
    return updated_artists

def plot_movements(sim, anim_length=10, anim_fps=12):
    """Animate the movements of all agents, including their sensing radius."""
    skip_frames = max(int(len(sim.history)/(anim_length * anim_fps)), 1)
    steps = np.array(sim.history[::skip_frames])  # Assuming sim.history is a list of positions for each step
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, sim.world.world_size)
    ax.set_ylim(0, sim.world.world_size)

    agent_circles = []  # This will store tuples of (position_circle, sensing_circle) for each agent
    
    for agent in steps[0]:
        if agent.agent_type == 'A':
            color = 'red'
        elif agent.agent_type == 'B':
            color = 'blue'
        else:
            color = 'green'
        position_circle = Circle(agent.position, radius=agent.radius, color=color)
        ax.add_patch(position_circle)
        
        # We can always add the sensing circle because radius=0 if we don't care about it (and you don't see it)
        sensing_circle = Circle(agent.position, radius=agent.sensing_radius, color=color, fill=False, linestyle='--')
        ax.add_patch(sensing_circle)
        
        agent_circles.append((position_circle, sensing_circle))
    
    anim = FuncAnimation(fig, animate_agents, frames=len(steps), interval=1e3/anim_fps, fargs=(agent_circles, steps), blit=True)
    
    ax.set_aspect('equal')

    # Create legend
    red_patch = Patch(color='red', label='A')
    blue_patch = Patch(color='blue', label='B')
    ax.legend(handles=[red_patch, blue_patch])

    plt.show()
    return anim

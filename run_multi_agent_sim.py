import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.simulator import Simulator
from src.world import World
from src.plot_agents import plot_movements
from src.helper_functions import create_agents
from src.plotting import plot_density, plot_rotated_histogram, plot_separation_index, plot_target_distance

def main(world, plot=False):
    #################
    # Create agents #
    #################
    agents = []
    agent_radius = 10

    # Create team A
    num_A_agents = 6
    step_length = 1      # np.ones((num_A_agents,))
    sensing_radius = 200 # 100*np.ones((num_A_agents,))
    create_agents(world, agents,
                  'A', num_A_agents, step_length, agent_radius, sensing_radius)

    # Create team B
    num_B_agents = 6
    step_length = 1
    sensing_radius = 200
    create_agents(world, agents,
                  'B', num_B_agents, step_length, agent_radius, sensing_radius)
    
    # Create team T
    num_T_agents = 0
    step_length = 1
    sensing_radius = 0
    create_agents(world, agents,
                  'T', num_T_agents, step_length, agent_radius, sensing_radius)
    
    ##################
    # Run simulation #
    ##################
    steps = 5000
    scenario = "b" # "b"

    sim = Simulator(world,agents,scenario)
    sim.simulate(steps=steps)
    if plot:
        plot_movements(sim, anim_length=10, anim_fps=24)

    return sim

if __name__ == "__main__":
    ################
    # Create world #
    ################
    world_limits = [0,1000]
    world = World(world_limits)

    ####################
    # Time-lapse plots #
    ####################
    sim = main(world, plot=False)

    fig, axs = plt.subplots(2,5, figsize=(15,6))
    times = np.linspace(0, len(sim.history)-1, 10).astype(int)
    times = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4999]
    for i, ax in enumerate(axs.flat):
        agents_snapshot = sim.history[int(times[i])]
        for agent in agents_snapshot:
            ax.plot(agent.position[0], agent.position[1], 'o', color='blue' if agent.agent_type=='A' else 'red')
        # ax.set_xlabel("x position")
        # ax.set_ylabel("y position")
        ax.set_xlim(world.x_lim)
        ax.set_ylim(world.y_lim)
        ax.grid()
        ax.set_title(f"t={times[i]}")
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])   
        ax.set_yticks([])
    plt.savefig("figures/time_lapse.png", dpi=300)

    ###########################
    # Run several simulations #
    ###########################
    sims = []
    # create tqdm
    N = 10
    pbar = tqdm(total=N)
    for i in range(N):
        sims.append(main(world))
        pbar.update(1)
    pbar.close()

    ############
    # Plotting #
    ############
    plot_density(sims, name="density")
    plot_rotated_histogram(sims, name="histogram")
    plot_separation_index(sims, name="orderliness")
    plot_target_distance(sims, name="target_distance")
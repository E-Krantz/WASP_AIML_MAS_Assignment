from src.simulator import Simulator
from src.world import World
from src.plot_agents import plot_movements
from src.helper_functions import create_agents

def main():
    ################
    # Create world #
    ################
    world_size = 250.0
    world = World(world_size)

    #################
    # Create agents #
    #################
    agents = []
    agent_radius = 5

    # Create team A
    num_A_agents = 5
    step_length = 1     # np.ones((num_A_agents,))
    sensing_radius = 100  # 100*np.ones((num_A_agents,))
    create_agents(world, agents,
                  'A', num_A_agents, step_length, agent_radius, sensing_radius)

    # Create team B
    num_B_agents = 5
    step_length = 1
    sensing_radius = 100
    create_agents(world, agents,
                  'B', num_B_agents, step_length, agent_radius, sensing_radius)
    
    # Create team T
    num_T_agents = 0
    step_length = 1
    sensing_radius = 100
    create_agents(world, agents,
                  'T', num_T_agents, step_length, agent_radius, sensing_radius)
    
    ##################
    # Run simulation #
    ##################
    steps = 1000

    sim = Simulator(world,agents)
    sim.simulate(steps=steps)
    plot_movements(sim, anim_length=10, anim_fps=24)

if __name__ == "__main__":
    main()
from src.simulator import Simulator
from src.world import World
from src.plot_agents import plot_movements
from src.helper_functions import create_agents

def main():
    ################
    # Create world #
    ################
    world_size = 1000.0
    world = World(world_size)

    #################
    # Create agents #
    #################
    agents = []
    agent_radius = 10

    # Create team A
    num_A_agents = 4
    step_length = 1     # np.ones((num_A_agents,))
    sensing_radius = 200 # 100*np.ones((num_A_agents,))
    create_agents(world, agents,
                  'A', num_A_agents, step_length, agent_radius, sensing_radius)

    # Create team B
    num_B_agents = 4
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
    steps = 10000
    scenario = "b" # "b"

    sim = Simulator(world,agents,scenario)
    sim.simulate(steps=steps)
    plot_movements(sim, anim_length=10, anim_fps=24)

if __name__ == "__main__":
    main()
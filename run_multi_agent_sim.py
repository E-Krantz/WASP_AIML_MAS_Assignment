from src.simulator import Simulator
from src.plot_agents import plot_movements

num_A_agents = 5
num_B_agents = 5
num_T_agents = 0
world_size = 250.0
steps = 1000
step_length = 1
agent_radius = 5
sensing_radius = 100

def main():
    sim = Simulator(world_size=world_size, num_A_agents=num_A_agents, num_B_agents=num_B_agents, num_T_agents=num_T_agents, 
                    step_length=step_length, agent_radius=agent_radius, sensing_radius=sensing_radius)
    sim.simulate(steps=steps)
    plot_movements(sim, anim_length=10, anim_fps=24)

if __name__ == "__main__":
    main()
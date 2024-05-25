import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures

from src.simulator import Simulator
from src.world import World
from src.plot_agents import plot_movements
from src.helper_functions import create_agents
from src.plotting import plot_density, plot_rotated_histogram, plot_separation_index, plot_target_distance

def plot_sims(sim_results,
              scenario,
              num_A_agents, num_B_agents,
              step_length_A, step_length_B,
              sensing_radius_A, sensing_radius_B):
    # Create part of the save string
    save_str = f"_{scenario}_numA{num_A_agents}_numB{num_B_agents}_stepA{step_length_A}_stepB{step_length_B}_senseA{sensing_radius_A}_senseB{sensing_radius_B}"
    title = f" $N_A: {num_A_agents}, N_A: {num_B_agents}, L_A: {step_length_A}, L_B: {step_length_B}, r_A: {sensing_radius_A}, r_B: {sensing_radius_B}$"
    # print("save_string:", save_str)
    # print("title: ", title)

    # plot_density(sim_results, title=f"Density with {title}", name=f"density{save_str}")
    # plot_rotated_histogram(sim_results, title=f"Histogram with {title}", name=f"histogram{save_str}")
    plot_separation_index(sim_results, title=f"Orderliness with {title}", name=f"orderliness{save_str}")
    # plot_target_distance(sim_results, title=f"Target distance with {title}", name=f"target_distance{save_str}")


def run_sims(world,
             num_sims,
             scenario,
             num_A_agents, num_B_agents,
             step_length_A, step_length_B,
             sensing_radius_A, sensing_radius_B):
    
    sim_results = []
    for _ in range(num_sims):
        #################
        # Create agents #
        #################
        agents = []
        agent_radius = 10

        create_agents(world, agents,
                    'A', num_A_agents, step_length_A, agent_radius, sensing_radius_A)

        create_agents(world, agents,
                    'B', num_B_agents, step_length_B, agent_radius, sensing_radius_B)
        
        ##################
        # Run simulation #
        ##################
        steps = 5000

        sim = Simulator(world,agents,scenario)
        sim.simulate(steps=steps)
        sim_results.append(sim)

    plot_sims(sim_results,
              scenario=scenario,
              num_A_agents=num_A_agents, num_B_agents=num_B_agents,
              step_length_A=step_length_A, step_length_B=step_length_B,
              sensing_radius_A=sensing_radius_A, sensing_radius_B=sensing_radius_B)

    return 1


if __name__ == "__main__":
    ####################
    # Create Scenarios #
    ####################
    world_limits = [0,1000]
    world = World(world_limits)

    num_sims = 50

    scenarios = []
    scenarios.append((world, num_sims, "a", 6, 6, 1, 1, 200, 200))
    scenarios.append((world, num_sims, "a", 6, 6, 2, 1, 200, 200))
    scenarios.append((world, num_sims, "a", 6, 6, 1, 1, 1000, 1000))
    scenarios.append((world, num_sims, "a", 3, 9, 1, 1, 200, 200))

    scenarios.append((world, num_sims, "b", 6, 6, 1, 1, 200, 200))
    scenarios.append((world, num_sims, "b", 6, 6, 2, 1, 200, 200))
    scenarios.append((world, num_sims, "b", 6, 6, 1, 1, 1000, 1000))
    scenarios.append((world, num_sims, "b", 3, 9, 1, 1, 200, 200))

    print("Created scenarios!")

    ############
    # Run sims #
    ############ 
    pbar = tqdm(total=len(scenarios))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run_sims, *scenarios[i]) for i in range(len(scenarios))]

        for future in concurrent.futures.as_completed(results):
            pbar.update(1)
    pbar.close()

    print("All simulations completed!")
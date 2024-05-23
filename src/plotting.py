import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.simulator import Simulator
from src.world import World
from src.plot_agents import plot_movements
from src.helper_functions import create_agents, pos_to_grid, get_density, \
    get_separation_index, circle_around_index, get_mean_target_distance
   
def plot_density(sims):
    ###############
    # Density Fig #
    ###############
    mean_pos_teams = []
    for sim in sims:
        # compute the mean position of each team
        mean_pos_teams.append(get_density(sim,rotate=False))
    mean_pos_teams = np.array(mean_pos_teams)

    # Create a heatmap of the mean position of each team
    # Create a grid of worldsize and add an occurrence of 1 at each position of the mean position of each team
    grid_world = World([-100,100])
    density = np.zeros((int(grid_world.x_lim[-1]-grid_world.x_lim[0]), 
                        int(grid_world.y_lim[-1]-grid_world.y_lim[0])))
    
    for it,data in enumerate(mean_pos_teams):
        # team A -> +1
        x,y = pos_to_grid(data[0], density, grid_world)
        xs,ys = circle_around_index(x,y, 5, density)
        for x,y in zip(xs,ys):
            density[x,y] += 1
        # team B -> -1
        x,y = pos_to_grid(data[1], density, grid_world)
        xs,ys = circle_around_index(x,y, 5, density)
        for x,y in zip(xs,ys):
            density[x,y] -= 1

    # Plot the heatmap
    fig, ax = plt.subplots()
    ax.imshow(density.T, cmap='coolwarm', interpolation='hermite',
              extent=[grid_world.x_lim[0], grid_world.x_lim[1], grid_world.y_lim[0], grid_world.y_lim[1]])
    # change the x and y ticks to match grid_world
    x_range = np.linspace(grid_world.x_lim[0], grid_world.x_lim[1], density.shape[0])
    y_range = np.linspace(grid_world.y_lim[0], grid_world.y_lim[1], density.shape[1])
    ax.set_xticks(np.arange(grid_world.x_lim[0], grid_world.x_lim[1]+0.1, 20))
    ax.set_yticks(np.arange(grid_world.y_lim[0], grid_world.y_lim[1]+0.1, 20))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.savefig("figures/density.png", dpi=300)

def plot_rotated_histogram(sims):
    #####################
    # Rotated Histogram #
    #####################
    # here we get the mean density of both teams, and rotate the mean position of team A to the x-axis
    # as the agents always converge to a line, we can plot this then on a 1D histogram
    mean_pos_teams = []
    for sim in sims:
        # compute the mean position of each team
        mean_pos_teams.append(get_density(sim,rotate=True))
    mean_pos_teams = np.array(mean_pos_teams)

    # indexing is [simulation, team, x/y]
    mean_x_pos_teams = mean_pos_teams[:,:,0]
    mean_x_pos_A = mean_x_pos_teams[:,0]
    mean_x_pos_B = mean_x_pos_teams[:,1]

    # Create a histogram of the mean position of each team
    fig, axs = plt.subplots(1,1, figsize=(12,6))
    axs.hist(mean_x_pos_A, bins=20, alpha=0.5, color='blue', label='Team A')
    axs.hist(mean_x_pos_B, bins=20, alpha=0.5, color='red', label='Team B')
    axs.set_xlabel("Mean x position [m]")
    axs.set_ylabel("Frequency [-]")
    axs.legend()
    plt.savefig("figures/histogram.png", dpi=300)

def plot_separation_index(sims):
    ####################
    # Seperation index #
    ####################
    # here we plot the separation index of each simulation
    # orderliness = avg_inter_team_distance / (avg_intra_team_distance_A + avg_intra_team_distance_B)/2
    # Get the separation of all sims over time
    orderlinesses = []
    intra_team_distances = []
    for sim in sims:
        indices = []
        distances = []
        # we do this for each sim, loop through the history and get the separation index
        for sim_state in sim.history:
            orderliness, total_intra_team_distance = get_separation_index(sim_state)
            indices.append(orderliness)
            distances.append(total_intra_team_distance)
        orderlinesses.append(indices)
        intra_team_distances.append(distances)
    
    # # Now plot the orderliness over time
    # fig, axs = plt.subplots(1,1, figsize=(12,6))
    # for sim in orderlinesses:
    #     axs.plot(sim)
    # axs.set_xlabel("Time")
    # axs.set_ylabel("Orderliness index")
    # plt.savefig("figures/orderliness_index.png", dpi=300)

    # # Now plot the intra team distances over time
    # fig, axs = plt.subplots(1,1, figsize=(12,6))
    # for sim in intra_team_distances:
    #     axs.plot(sim)
    # axs.set_xlabel("Time")
    # axs.set_ylabel("Intra team distances")
    # plt.savefig("figures/intra_team_distances_index.png", dpi=300)

    # Create a plot with the mean and standard deviation over time for all the different simulations
    mean_orderlinesses_index = np.mean(orderlinesses, axis=0)
    std_orderlinesses_index = np.std(orderlinesses, axis=0)

    mean_intra_team_distances_index = np.mean(intra_team_distances, axis=0)
    std_intra_team_distances_index = np.std(intra_team_distances, axis=0)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(mean_orderlinesses_index)
    axs[0].fill_between(range(len(mean_orderlinesses_index)), 
                     mean_orderlinesses_index-std_orderlinesses_index, 
                     mean_orderlinesses_index+std_orderlinesses_index, alpha=0.5)
    axs[0].set_xlabel("Time [it]")
    axs[0].set_ylabel("Separation index [m]")

    axs[1].plot(mean_intra_team_distances_index)
    axs[1].fill_between(range(len(mean_intra_team_distances_index)), 
                     mean_intra_team_distances_index-std_intra_team_distances_index, 
                     mean_intra_team_distances_index+std_intra_team_distances_index, alpha=0.5)
    axs[1].set_xlabel("Time [it]")
    axs[1].set_ylabel("Intra team distances [m]")

    plt.savefig("figures/indices_mean_std.png", dpi=300)

def plot_target_distance(sims):
    #####################
    # Target distance #
    #####################
    # here we plot the target distance of each simulation
    # Get the target distance of all sims over time
    mean_target_distances_A = []
    mean_target_distances_B = []
    for sim in sims:
        distances_A = []
        distances_B = []
        # we do this for each sim, loop through the history and get the separation index
        for sim_state in sim.history:
            mean_target_distance_A, mean_target_distance_B = get_mean_target_distance(sim_state)
            distances_A.append(mean_target_distance_A)
            distances_B.append(mean_target_distance_B)
        mean_target_distances_A.append(distances_A)
        mean_target_distances_B.append(distances_B)
    
    # Create a plot with the mean and standard deviation over time for all the different simulations
    mean_target_distances_A = np.mean(mean_target_distances_A, axis=0)
    std_target_distances_A = np.std(mean_target_distances_A, axis=0)

    mean_target_distances_B = np.mean(mean_target_distances_B, axis=0)
    std_target_distances_B = np.std(mean_target_distances_B, axis=0)

    fig, axs = plt.subplots(1,1, figsize=(12,6))
    axs.plot(mean_target_distances_A)
    axs.fill_between(range(len(mean_target_distances_A)), 
                     mean_target_distances_A-std_target_distances_A, 
                     mean_target_distances_A+std_target_distances_A, alpha=0.5)
    
    axs.plot(mean_target_distances_B)
    axs.fill_between(range(len(mean_target_distances_B)), 
                     mean_target_distances_B-std_target_distances_B, 
                     mean_target_distances_B+std_target_distances_B, alpha=0.5)
    
    axs.set_xlabel("Time [it]")
    axs.set_ylabel("Mean target distance [m]")

    plt.savefig("figures/target_distance_mean_std.png", dpi=300)
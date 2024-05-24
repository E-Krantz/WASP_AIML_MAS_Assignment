from src.agent import Agent
from src.world import World
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def create_agents(world,agents,
                  agent_type, num_agents, step_length, agent_radius, sensing_radius):
    # assert that sensing_radius is either a single value or an numpy array of length num_agents
    if not isinstance(sensing_radius, np.ndarray):
        sensing_radius = sensing_radius * np.ones((num_agents,))
    assert len(sensing_radius) == num_agents, "sensing_radius must be a single value or a numpy array of length num_agents"

    # assert that step_length is either a single value or an numpy array of length num_agents
    if not isinstance(step_length, np.ndarray):
        step_length = step_length * np.ones((num_agents,))
    assert len(step_length) == num_agents, "step_length must be a single value or a numpy array of length num_agents"

    # then add an agent on a position that is not yet occupied
    for i in range(num_agents):
            position_valid = False
            while not position_valid:
                position = np.random.rand(2) * (world.world_size - 2*agent_radius) + agent_radius
                if not check_overlap(position,agents):
                    position_valid = True
                    agents.append(Agent(agent_type, position, 
                                        step_length=step_length[i], radius=agent_radius, sensing_radius=sensing_radius[i],
                                        world_size=world.world_size))
    return agents

def check_overlap(position,agents):
    """Check if the new agent overlaps with existing agents."""
    for other in agents:
        if np.linalg.norm(position - other.position) < 2*other.radius:
            return True  # Overlap found
    return False  # No overlap

def pos_to_grid(position, grid, grid_world):
    """Convert continuous position to a grid position in the world grid"""
    # We have the original world, but also a plotting world for the mean position of the agents!
    x_range = np.linspace(grid_world.x_lim[0], grid_world.x_lim[1], grid.shape[0])
    y_range = np.linspace(grid_world.y_lim[0], grid_world.y_lim[1], grid.shape[1])

    x = np.argmin(np.abs(x_range - position[0]))
    y = np.argmin(np.abs(y_range - position[1]))

    return x, y

def get_density(sim, rotate=False, plot=False):
    # Compute the relative density of the final state of all agents.
    # We center the density around the origin, and rotate it so that the
    # density of type 'A' agent is horizontal.
    agents_A = [agent for agent in sim.agents if agent.agent_type == 'A']
    agents_B = [agent for agent in sim.agents if agent.agent_type == 'B']

    mean_pos_A = np.mean([agent.position for agent in agents_A], axis=0)
    mean_pos_B = np.mean([agent.position for agent in agents_B], axis=0)
    mean_pos_All = np.mean([agent.position for agent in sim.agents], axis=0)

    mean_pos_A = mean_pos_A - mean_pos_All
    mean_pos_B = mean_pos_B - mean_pos_All

    if rotate:
        # get rotation matrix such that mean_pos_A is on the negative x-axis
        angle = np.pi - np.arctan2(mean_pos_A[1], mean_pos_A[0])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        
        # rotate the mean positions
        mean_pos_A = R.dot(mean_pos_A)
        mean_pos_B = R.dot(mean_pos_B)

    if plot:
        axs, fig = plt.subplots(1, 1, figsize=(6, 6))
        # plot the position of each agent
        for agent in sim.agents:
            color = 'blue' if agent.agent_type == 'A' else 'red'
            plt.plot(agent.position[0], agent.position[1], 'o', color=color)
        # plot the density of each team
        plt.plot(mean_pos_A[0], mean_pos_A[1], 'o', color='blue', markersize=10)
        plt.plot(mean_pos_B[0], mean_pos_B[1], 'o', color='red', markersize=10)
        plt.show()

    return [mean_pos_A, mean_pos_B]

def get_mean_target_distance(sim_instance):
    agents_A = [agent for agent in sim_instance if agent.agent_type == 'A']
    agents_B = [agent for agent in sim_instance if agent.agent_type == 'B']

    mean_target_distance_A = np.mean([agent.target_distance for agent in agents_A])
    mean_target_distance_B = np.mean([agent.target_distance for agent in agents_B])

    return mean_target_distance_A, mean_target_distance_B

def get_separation_index(sim_instance):
    agents_A = [agent for agent in sim_instance if agent.agent_type == 'A']
    agents_B = [agent for agent in sim_instance if agent.agent_type == 'B']

    positions_A = np.array([agent.position for agent in agents_A])
    positions_B = np.array([agent.position for agent in agents_B])
    positions = np.concatenate([positions_A, positions_B], axis=0)

    # Compute the distance between all agents
    pairwise_distances = squareform(pdist(positions))
    intra_A = pairwise_distances[:len(positions_A), :len(positions_A)]
    intra_B = pairwise_distances[len(positions_A):, len(positions_A):]

    avg_inter_team_distance = np.mean(pairwise_distances)
    # only take the average over the upper triangle of the intra_A matrix
    avg_intra_team_A_distance = np.mean(intra_A[np.triu_indices(len(positions_A), k=1)])
    avg_intra_team_B_distance = np.mean(intra_B[np.triu_indices(len(positions_B), k=1)])

    orderliness = avg_inter_team_distance / ((avg_intra_team_A_distance + avg_intra_team_B_distance) / 2)
    total_intra_team_distance = avg_intra_team_A_distance + avg_intra_team_B_distance
    return orderliness, total_intra_team_distance

def circle_around_index(x,y, r, density):
    # input is x and y which is an index in an array
    # we want to return a set of indices xs and ys that are within a circle of radius r
    # around the index x,y
    xs = []
    ys = []
    for i in range(max(0, x-r),min(density.shape[0], x+r)):
        for j in range(max(0, y-r),min(density.shape[1]-1, y+r)):
            if np.sqrt((i-x)**2 + (j-y)**2) < r:
                xs.append(i)
                ys.append(j)
    return xs,ys
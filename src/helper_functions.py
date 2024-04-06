from src.agent import Agent
from src.world import World
import numpy as np

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
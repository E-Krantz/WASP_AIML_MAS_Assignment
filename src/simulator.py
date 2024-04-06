import numpy as np
from .agent import Agent

class Simulator:
    def __init__(self, world, agents):
        self.world = world
        self.agents = agents
        self.history = []
    
    def update(self):
        # Update positions of all agents for one simulation step.
        for agent in self.agents:
            other_agents = [other for other in self.agents if other != agent]
            if agent.sensing_radius > 0: # 0 indicates we always sense other agents!
                other_agents = [other for other in other_agents if np.linalg.norm(other.position - agent.position) < agent.sensing_radius]
            # agent.update_target_position(other_agents)
            agent.move(other_agents)
        
        # Store the current positions of all agents
        self.history.append([(agent.position[0], agent.position[1]) for agent in self.agents])
    
    def simulate(self, steps):
        # Run the simulation for a given number of steps.
        for _ in range(steps):
            self.update()
            # Add progress bar
            print(f"Simulation progress: {len(self.history)}/{steps}", end="\r")

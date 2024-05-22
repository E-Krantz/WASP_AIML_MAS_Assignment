import numpy as np
from .agent import Agent
import matplotlib.pyplot as plt
import copy

class Simulator:
    def __init__(self, world, agents, scenario="a"):
        self.world = world
        self.agents = agents
        self.scenario = scenario
        self.history = []
    
    def update(self):
        # Update positions of all agents for one simulation step.
        for agent in self.agents:
            other_agents = [other for other in self.agents if other != agent]
            agent.move(other_agents,self.scenario)
        
        # Store the current positions of all agents
        # self.history.append([(agent.position[0], agent.position[1]) for agent in self.agents])
        # Store the current positions of all the agents by making a deep copy of the agent classes at this time instance
        self.history.append([copy.deepcopy(agent) for agent in self.agents])
    
    def simulate(self, steps):
        # Run the simulation for a given number of steps.
        for _ in range(steps):
            self.update()
            # Add progress bar
            # print(f"Simulation progress: {len(self.history)}/{steps}", end="\r")

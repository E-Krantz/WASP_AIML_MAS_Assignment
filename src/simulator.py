import numpy as np
from .agent import Agent

class Simulator:
    def __init__(self, world_size, num_A_agents, num_B_agents, num_T_agents, step_length=1, agent_radius=100):
        self.world_size = world_size
        self.agent_radius = agent_radius
        self.step_length = step_length
        self.agents = []
        self.history = []

        for _ in range(num_A_agents):
            position_valid = False
            while not position_valid:
                position = np.random.rand(2) * (world_size - 2*agent_radius) + agent_radius
                if not self.check_overlap(position):
                    position_valid = True
                    self.agents.append(Agent('A', position, step_length=self.step_length, radius=agent_radius, world_size=world_size))
        
        for _ in range(num_B_agents):
            position_valid = False
            while not position_valid:
                position = np.random.rand(2) * (world_size - 2*agent_radius) + agent_radius
                if not self.check_overlap(position):
                    position_valid = True
                    self.agents.append(Agent('B', position, step_length=self.step_length, radius=agent_radius, world_size=world_size))

        for _ in range(num_T_agents):
            position_valid = False
            while not position_valid:
                position = np.random.rand(2) * (world_size - 2*agent_radius) + agent_radius
                if not self.check_overlap(position):
                    position_valid = True
                    self.agents.append(Agent('T', position, step_length=self.step_length, radius=agent_radius, world_size=world_size))
    
    def update(self):
        # Update positions of all agents for one simulation step.
        for agent in self.agents:
            other_agents = [other for other in self.agents if other != agent]
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

    def check_overlap(self, new_agent_pos):
        """Check if the new agent overlaps with existing agents."""
        for agent in self.agents:
            if np.linalg.norm(new_agent_pos - agent.position) < 2*self.agent_radius:
                return True  # Overlap found
        return False  # No overlap
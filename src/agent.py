import numpy as np

class Agent:
    collision = False
    def __init__(self, agent_type, position, step_length=1, 
                 radius=100, sensing_radius=200,
                 world_size=1000):
        self.agent_type = agent_type
        self.position = np.array(position)
        self.target_position = np.random.rand(2) * world_size
        self.step_length = step_length
        self.radius = radius
        self.sensing_radius = sensing_radius
        self.world_size = world_size
        self.avoidance_direction = 1 if agent_type == 'A' else -1 #np.sign(np.random.rand() - 0.5)

    def move(self, other_agents, scenario):
        self.update_target_position(other_agents, scenario)
        
        """ Move towards a target position if it's not the current position.
        If new position will cause a collision, rotate direction vector (move around)."""
        if not np.array_equal(self.target_position, self.position):
            direction0 = (self.target_position - self.position)
            distance = np.linalg.norm(direction0)
            step_length = min(distance, self.step_length)
            direction0 = direction0 / distance
            for theta in range(0, 330, 30):
                direction = rotate_vector(direction0, self.avoidance_direction*theta)
                new_position = self.position + step_length * direction
            
                if not self.will_collide(new_position, other_agents):
                    self.position = new_position
                    break
                step_length = self.step_length

    def update_target_position(self, other_agents, scenario):
        # Find a target position for the agent by locating the nearest opposite type agent.
        A_list = []
        B_list = []
        target_position = self.target_position
        closest_point = float('inf') * np.ones(2)
        offset = 4*self.radius # some extra offset to ensure we are not too close to A or B
        """Find all agents of type A and B.
        We have included a perception radius too."""
        for other in other_agents:
            if self.sensing_radius == 0 or np.linalg.norm(self.position - other.position) < self.sensing_radius:
                if other.agent_type == 'A':
                    A_list.append(other)
                elif other.agent_type == 'B':
                    B_list.append(other)
        
        other_agents = A_list + B_list

        
        if (A_list == [] or B_list == []) and np.linalg.norm(self.target_position - self.position) < self.step_length:
            target_position = np.random.rand(2) * (self.world_size-4*self.radius) + 2*self.radius
        elif A_list != [] and B_list != []:
            """Find the closest point between A and B.
            First check projection of A and B, if this point
            is outside AB, then find the closest point is either
            A or B (minus 2*radius)."""
            dist_to_target = float('inf')
            valid_target = False
            for agent_A in A_list:
                for agent_B in B_list:
                    if dist_to_target > 0:
                        A = agent_A.position
                        B = agent_B.position
                        C = self.position
                        AB = B - A
                        AC = C - A
                        BC = C - B
                        proj_scalar = np.dot(AC, AB) / np.dot(AB, AB)
                        D = A + proj_scalar * AB

                        if scenario == 'a':
                            if np.linalg.norm(AB) > 4*self.radius:
                                # Check if the projection scalar is between 0 and 1
                                if 0 <= proj_scalar <= 1:
                                    closest_point = D
                                else:
                                    # Determine if A or B is closer to C
                                    if np.linalg.norm(AC) < np.linalg.norm(BC):
                                        closest_point = A + offset * AB/np.linalg.norm(AB)
                                    else:
                                        closest_point = B + offset * -AB/np.linalg.norm(AB)

                        elif scenario == 'b':
                            # The agent can either place itself so that B is between A and yourself (this is colsest_point_0),
                            # or so that B is between A and a target agent (this is closest_point).
                            # Choose the closest of these two points.
                            # if self.agent_type == 'A':
                            #     # ensure that B is inbetween self and A
                            #     # so take the vector AB, be sure that target_position is on this line, but extend it slightly to 
                            #     # ensure we are further than B is on this line
                            #     closest_point_0 = B + (2*self.radius+offset) * AB/np.linalg.norm(AB)             
                            # elif self.agent_type == 'B':
                            #     # ensure that A is inbetween self and B
                            #     closest_point_0 = A + (2*self.radius+offset) * -AB/np.linalg.norm(AB)
                            # else:
                            #     closest_point_0 = closest_point

                            # Check if the projection scalar is outside A and B
                            if proj_scalar > 1:
                                closest_point = D
                            else:
                                # ensure that B is inbetween self and A
                                # so take the vector AB, be sure that target_position is on this line, but extend it slightly to 
                                # ensure we are further than B is on this line
                                closest_point = B + offset * AB/np.linalg.norm(AB)
                            
                            # if np.linalg.norm(closest_point_0 - self.position) < np.linalg.norm(closest_point - self.position):
                            #     closest_point = closest_point_0

                        if np.linalg.norm(closest_point - self.position) < 1e-1 * self.step_length:
                            dist_to_target = 0
                            target_position = self.position
                            valid_target = True                         
                            
                        elif np.linalg.norm(closest_point - self.position) < dist_to_target and not self.will_collide(closest_point, other_agents):
                            dist_to_target = np.linalg.norm(closest_point - self.position)
                            target_position = closest_point
                            valid_target = True
                            
            while not valid_target:
                target_position = np.random.rand(2) * (self.world_size-4*self.radius) + 2*self.radius
                valid_target = not self.will_collide(target_position, other_agents)

        # Ensure that the target position is within the world boundaries
        target_position[0] = max(target_position[0], self.radius*1.1)
        target_position[0] = min(target_position[0], self.world_size-self.radius*1.1)
        target_position[1] = max(target_position[1], self.radius*1.1)
        target_position[1] = min(target_position[1], self.world_size-self.radius*1.1)

        # Update the target position
        self.target_position = target_position

    def will_collide(self, new_position, other_agents):
        if new_position[0] < self.radius or new_position[0] > self.world_size-self.radius or new_position[1] < self.radius or new_position[1] > self.world_size-self.radius:
            return True
        for other in other_agents:
            if other != self:
                distance = np.linalg.norm(new_position - other.position)
                if distance < (self.radius + other.radius):
                    return True
        return False

def rotate_vector(vector, angle):
    """Rotate a 2D vector by a given angle."""
    angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, vector)
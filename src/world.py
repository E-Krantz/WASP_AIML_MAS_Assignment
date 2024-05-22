class World:
    def __init__(self,bounds):
        self.world_size = bounds[-1] - bounds[0]

        self.x_lim = bounds
        self.y_lim = bounds
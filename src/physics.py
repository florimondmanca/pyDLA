import numpy as np
from scipy.spatial.distance import cdist


class DLA:

    def __init__(self, number_of_particles, size, agg_r=5,
                 temperature=5, no_centers=False):
        self.nb = number_of_particles
        self.size = size
        self.box = (size, size)
        particles = np.zeros(self.nb,
                             dtype=[('pos', ('f', 2)), ('vel', ('f', 2))])
        x = self.size * np.random.random(self.nb)
        y = self.size * np.random.random(self.nb)
        particles['pos'] = np.column_stack((x, y))
        # aggregation array. it is True for particles that aggregated
        # and won't move in the future
        self.agg_array = np.zeros(self.nb, dtype=np.bool)
        particles['vel'] = self.randvel()
        self.particles = particles
        self.agg_r = agg_r
        self.temperature = temperature
        if not no_centers:
            self.centers = np.array([[self.size / 2, self.size / 2]])
        else:
            self.centers = []

    def add_center(self, center):
        if self.centers:
            self.centers = np.vstack((self.centers, center))
        else:
            self.centers = np.array(center).reshape((1, 2))

    def randvel(self):
        vel = np.random.randn(self.nb, 2)
        vel[self.agg_array] = 0  # aggregated particles don't move anymore
        return vel

    def move(self):
        # generate new velocities
        self.particles['vel'] = self.randvel() * self.temperature
        # move
        self.particles['pos'] += self.particles['vel']
        p = self.particles['pos']
        # constrain in box
        p[p < 0] += self.size
        p[p > self.size] -= self.size
        # aggregate:
        # 1° compute distances from non-aggregated points to aggregated points
        nonagged = p[~self.agg_array]
        agged = np.vstack((p[self.agg_array], self.centers))
        dist = cdist(nonagged, agged)
        # 2° aggregate those near to aggregation points
        whereagg = np.zeros(self.agg_array.shape, dtype=np.bool)
        whereagg[~self.agg_array] = np.any(dist < self.agg_r, axis=1)
        self.agg_array[whereagg] = True
        # particles['vel'][agg] = 0

    def fixed_particles(self):
        return self.particles[self.agg_array]

    def all_fixed(self):
        return np.all(self.agg_array)

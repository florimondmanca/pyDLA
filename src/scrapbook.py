import numpy as np
from scipy.spatial.distance import cdist

n = 10  # particles
# random positions
pos = np.random.random((n, 2))
# random aggregation
aggr = np.zeros(n, dtype=np.bool)
aggr[np.random.random(aggr.shape) < .5] = True
# calculate distance from center
d = cdist(pos, np.ones((2, 2)) / 2)
print(d)
# if inferior to some value, aggregate
aggr[np.any(d < 0.3, axis=1)] = True
# velocity generation
vel = (2 * np.random.random((n, 2)) - 1)
vel[aggr] = 0

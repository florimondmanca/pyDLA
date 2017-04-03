import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist


NPARTICLES = 100
L = 100
AGGR_R = 1  # aggregation radius
CENTER = np.ones((1, 2)) * L / 2


def randvel(aggr, temp=10):
    vel = (2 * np.random.random((NPARTICLES, 2)) - 1) * temp
    vel[aggr] = 0  # aggregated particles don't move anymore
    return vel


def move(particles, aggr):
    particles['pos'] += particles['vel']
    p = particles['pos']
    # constrain in box
    p[p < 0] += L
    p[p > L] -= L
    # aggregate to center :
    # 1° compute distances to center
    dist = cdist(p, CENTER)[:, 0]
    # 2° aggregate (freeze) those that are near to it
    aggr[dist < AGGR_R] = True

    # generate new velocities
    particles['vel'] = randvel(aggr)


def initp():
    particles = np.zeros(NPARTICLES,
                         dtype=[('pos', ('f', 2)), ('vel', ('f', 2))])
    x = L * np.random.random(NPARTICLES)
    y = L * np.random.random(NPARTICLES)
    particles['pos'] = np.column_stack((x, y))
    aggr = np.zeros(NPARTICLES, dtype=np.bool)
    particles['vel'] = randvel(aggr)
    return particles, aggr


def anim():
    # simulation initialization
    particles, aggr = initp()
    # plot initialization
    dt = 1 / 30  # fps
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, L), ylim=(0, L))
    ax.grid()
    line, = ax.plot([], [], 'bo', lw=2)

    def init():
        """initialize animation"""
        line.set_data([], [])
        return line,

    def animate(i):
        """perform animation step"""
        move(particles, aggr)
        x = particles['pos'][:, 0]
        y = particles['pos'][:, 1]
        line.set_data(x, y)
        return line,

    from time import time
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=int(10 / dt),
                                  interval=interval, blit=False,
                                  init_func=init)
    plt.show()


if __name__ == '__main__':
    anim()

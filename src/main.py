import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
from time import time


NPARTICLES = 1000
L = 500
AGGR_R = 7  # aggregation radius
CENTER = np.ones((1, 2)) * L / 2


def randvel(aggr, temp=5):
    vel = (2 * np.random.random((NPARTICLES, 2)) - 1) * temp
    vel[aggr] = 0  # aggregated particles don't move anymore
    return vel


def move(particles, aggr):
    # generate new velocities
    particles['vel'] = randvel(aggr)
    # move
    particles['pos'] += particles['vel']
    p = particles['pos']
    # constrain in box
    p[p < 0] += L
    p[p > L] -= L
    # aggregate:
    # 1° compute distances from points to aggregation points
    aggr_points = np.vstack((p[aggr], CENTER))
    dist = cdist(p, aggr_points)
    # 2° aggregate those near to aggregation points
    aggr[np.any(dist < AGGR_R, axis=1)] = True


def initp():
    particles = np.zeros(NPARTICLES,
                         dtype=[('pos', ('f', 2)), ('vel', ('f', 2))])
    x = L * np.random.random(NPARTICLES)
    y = L * np.random.random(NPARTICLES)
    particles['pos'] = np.column_stack((x, y))
    # aggregation matrix. it is True for particles that aggregated
    # and won't move in the future
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
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    moving, = ax.plot([], [], 'bo')
    fixed, = ax.plot([], [], 'ro')

    def init():
        """initialize animation"""
        moving.set_data([], [])
        fixed.set_data([], [])
        return moving, fixed

    def animate(i):
        """perform animation step"""
        t = time()
        move(particles, aggr)
        moving_p = particles['pos'][~aggr]
        fixed_p = particles['pos'][aggr]
        moving.set_data(moving_p[:, 0], moving_p[:, 1])
        fixed.set_data(fixed_p[:, 0], fixed_p[:, 1])
        print(time() - t)
        return moving, fixed

    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=int(1 / dt),
                                  interval=interval, blit=False,
                                  init_func=init)
    plt.show()


if __name__ == '__main__':
    anim()

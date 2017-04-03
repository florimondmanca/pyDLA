import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
from time import time
import pygame


NPARTICLES = 1000
L = 400
AGGR_R = 6  # aggregation radius
CENTER = np.ones((1, 2)) * L / 2


def randvel(aggr):
    vel = (2 * np.random.random((NPARTICLES, 2)) - 1)
    vel[aggr] = 0  # aggregated particles don't move anymore
    return vel


def move(particles, aggr, temp=5):
    # generate new velocities
    particles['vel'] = randvel(aggr) * temp
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


def pganim():
    particles, aggr = initp()
    size = (L, L)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    fps = 60
    r = AGGR_R // 6  # particle radius on graph
    while True:
        t = time()
        screen.fill((255, 255, 255))
        clock.tick(fps)
        # update physics
        move(particles, aggr, temp=6)
        # plot the particles onto an image
        surf = np.ones((*size, 3)) * 255
        moving_p = particles['pos'][~aggr]
        for x, y in moving_p:
            a, b = int(x), int(y)
            surf[a - r:a + r, b - r:b + r] = (255, 0, 0)
        fixed_p = particles['pos'][aggr]
        for x, y in fixed_p:
            a, b = int(x), int(y)
            surf[a - r:a + r, b - r:b + r] = (0, 0, 0)
        # draw the array
        screen.blit(pygame.surfarray.make_surface(surf), (0, 0))
        pygame.display.flip()
        # print(1 / (time() - t))
        if pygame.event.peek(pygame.QUIT):
            print('exitting')
            break


if __name__ == '__main__':
    pganim()

import numpy as np
from scipy.spatial.distance import cdist
from time import time
import pygame


NPARTICLES = 2000
L = 500
AGGR_R = 8  # aggregation radius
centers = [(.5, .5), ]
CENTERS = L * np.vstack(centers)


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


def randvel(aggr):
    vel = np.random.randn(NPARTICLES, 2)
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
    # 1° compute distances from non-aggregated points to aggregated points
    nonagged = p[~aggr]
    agged = np.vstack((p[aggr], CENTERS))
    dist = cdist(nonagged, agged)
    # 2° aggregate those near to aggregation points
    whereagg = np.zeros(aggr.shape, dtype=np.bool)
    whereagg[~aggr] = np.any(dist < AGGR_R, axis=1)
    aggr[whereagg] = True
    # particles['vel'][agg] = 0


def pganim():
    particles, aggr = initp()
    size = (L, L)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    fps = 60
    r = 2  # particle radius on graph
    while True:
        clock.tick(fps)
        # update physics
        t = time()
        move(particles, aggr, temp=3)
        # print(1 / (time() - t))
        # plot the particles onto an image
        surf = np.zeros((L, L, 3))
        surf[:, :] = 255
        for pos, fixed in zip(particles['pos'], aggr):
            if fixed:
                x, y = map(int, pos)
                d = 1 - np.sqrt((x - L / 2)**2 + (y - L / 2)**2) / (L * .707)
                surf[x - r:x + r, y - r:y + r] = [255 * d, 100 * d, 100 * d]
        # draw the image
        screen.blit(pygame.surfarray.make_surface(surf), (0, 0))
        pygame.display.flip()
        if pygame.event.peek(pygame.QUIT):
            print('exitting')
            break
        if np.all(aggr):
            print('All particles aggregated.')
            while not pygame.event.peek(pygame.QUIT):
                clock.tick(fps)
            break


if __name__ == '__main__':
    pganim()

import numpy as np
from scipy.spatial.distance import cdist
from time import time
import pygame


NPARTICLES = 2000
L = 400
AGGR_R = 5  # aggregation radius
centers = [(.5, .5), (.2, .8), (.8, .4), (.7, .65), (0.2, .15)]
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
    angle = np.random.random(NPARTICLES) * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    vel = np.column_stack((c, s))
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
    aggr_points = np.vstack((p[aggr], CENTERS))
    dist = cdist(p, aggr_points)
    # 2° aggregate those near to aggregation points
    agg = np.any(dist < AGGR_R, axis=1)
    aggr[agg] = True
    particles['vel'][agg] = 0


def pganim():
    particles, aggr = initp()
    size = (L, L)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    fps = 30
    r = 1  # particle radius on graph
    while True:
        t = time()
        clock.tick(fps)
        # update physics
        move(particles, aggr, temp=6)
        # plot the particles onto an image
        surf = np.zeros((L, L, 3))
        surf[:, :] = 255
        for pos, fixed in zip(particles['pos'], aggr):
            if fixed:
                x, y = map(int, pos)
                surf[x - r:x + r, y - r:y + r] = [150, 120, 200]
        # draw the image
        screen.blit(pygame.surfarray.make_surface(surf), (0, 0))
        pygame.display.flip()
        # print(1 / (time() - t))
        if pygame.event.peek(pygame.QUIT):
            print('exitting')
            break


if __name__ == '__main__':
    pganim()

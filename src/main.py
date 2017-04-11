import numpy as np
import pygame
from scipy.signal import convolve2d


EMPTY = 0
FIXED = 1
MOBILE = 2
COLORS = {
    EMPTY: (0, 0, 0),
    FIXED: (255, 255, 255),
    # MOBILE: (100, 150, 150),
}
M = {
    'up': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    'down': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    'left': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    'right': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
}
DIR_ROLL = {
    'up': (-1, 1),
    'down': (1, 1),
    'left': (-1, 0),
    'right': (1, 0),
}
N_MASK = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])


def shapenize(shape):
    try:
        shape[0]
    except TypeError:
        shape = (shape, shape)
    return shape


def empty(shape):
    lattice = np.zeros(shape)
    return lattice


def make(shape, density=.25):
    shape = shapenize(shape)
    lattice = empty(shape)
    lattice[np.random.random(shape) < density] = MOBILE
    lattice[:, -1] = FIXED
    return lattice


def evolve(lattice):
    # mobile particles move
    mobile = lattice == MOBILE
    for d in M:
        shift, axis = DIR_ROLL[d]
        willmove_d = mobile * (np.random.random(lattice.shape) < .75)
        free_d = convolve2d(
            lattice, M[d], mode='same') == 0
        move_d = willmove_d * free_d
        lattice[move_d] = EMPTY
        lattice[np.roll(move_d, shift, axis)] = MOBILE
        mobile = lattice == MOBILE
    # fixed particles absorb mobile particles that are near neighbors
    fixed = lattice == FIXED
    has_fixed_ngbr = mobile * (
        convolve2d(fixed, N_MASK, mode='same') >= 1)
    lattice[has_fixed_ngbr] = FIXED
    return lattice


def zoomit(lattice, size=None, default_size=750):
    if size is None:
        size = default_size
    if size < max(lattice.shape):
        size = max(lattice.shape)
    zoom = size / max(lattice.shape)
    zoomed = np.repeat(np.repeat(lattice, zoom, axis=0), zoom, axis=1)
    return zoomed


def show(lattice, fps=30):
    # from time import time
    zoomed = zoomit(lattice)
    windowsize = zoomed.shape
    pygame.init()
    try:
        screen = pygame.display.set_mode(windowsize)
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(fps)
            # t = time()
            lattice = evolve(lattice)
            # print(1 / (time() - t))
            zoomed = zoomit(lattice)
            carr = np.zeros((*zoomed.shape, 3))
            for T, C in COLORS.items():
                carr[zoomed == T] = C
            pygame.surfarray.blit_array(screen, carr)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    finally:
        pygame.quit()


if __name__ == '__main__':
    show(make(400))

import numpy as np
import pygame
from scipy.spatial.distance import norm
from physics import DLA


def pganim(n, l, r, temp):
    dla = DLA(n, l, agg_r=r, temperature=temp)
    size = dla.box
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(
        'pyDLA |\t#P: {}  L: {}  R: {}  T: {}'
        .format(n, l, r, temp), 'pyDLA')
    clock = pygame.time.Clock()
    fps = 60
    r = 2  # particle radius on graph
    hs = dla.size / 2
    paused = False
    running = True
    while running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
            if event.type == pygame.QUIT:
                running = False
        if paused:
            continue
        # update physics
        dla.move()
        # plot the particles onto an image
        surf = np.zeros((*size, 3))
        surf[:, :] = 255
        for particle in dla.fixed_particles():
            x, y = map(int, particle['pos'])
            d = 1 - norm((x - hs, y - hs)) / (dla.size * .707)
            surf[x - r:x + r, y - r:y + r] = [255 * d, 100 * d, 100 * d]
        # draw the image
        screen.blit(pygame.surfarray.make_surface(surf), (0, 0))
        pygame.display.flip()
        if dla.all_fixed():
            print('All particles aggregated.')
            while not pygame.event.peek(pygame.QUIT):
                clock.tick(fps)
            break


if __name__ == '__main__':
    pganim(2000, 500, 8, 3)

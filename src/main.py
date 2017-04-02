import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


NPARTICLES = 100
L = 100


def randvel(temp=10):
    return (2 * np.random.random((NPARTICLES, 2)) - 1) * temp


def move(particles):
    particles['pos'] += particles['vel']
    # constrain in box
    p = particles['pos']
    p[p < 0] += L
    p[p > L] -= L
    # generate new velocities
    particles['vel'] = randvel()


def initp():
    particles = np.zeros(NPARTICLES,
                         dtype=[('pos', ('f', 2)), ('vel', ('f', 2))])
    x = L * np.random.random(NPARTICLES)
    y = L * np.random.random(NPARTICLES)
    particles['pos'] = np.column_stack((x, y))
    particles['vel'] = randvel()
    return particles


def anim():
    # simulation initialization
    particles = initp()
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
        move(particles)
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

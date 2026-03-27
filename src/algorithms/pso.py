import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, bounds, n):
        self.x = np.random.uniform(bounds[0], bounds[1], size=n)
        self.v = np.random.uniform(0.1 * bounds[0], 0.1 * bounds[1], size=n)
        self.best_x = self.x

def particle_swarm_opt(f, p_num, n, bounds, max_epochs=100, alph1=1, alph2=1):
    phi1 = np.random.uniform()
    phi2 = np.random.uniform()

    #phi1 = 0.5
    #phi2 = 0.5

    particles = [Particle(bounds, n) for _ in range(p_num)]
    scores = []

    gbest = particles[np.argmin([f(particle.x) for particle in particles])].best_x
    curr_epoch = 1

    while curr_epoch < max_epochs:
        for particle in particles:
            new_v = particle.v + alph1 * phi1 * (particle.best_x - particle.x) + alph2 * phi2 * (gbest - particle.x)
            new_x = particle.x+new_v

            if f(new_x) < f(particle.best_x):
                particle.best_x = new_x
            if f(new_x) < f(gbest):
                gbest = new_x
            
            particle.v = new_v
            particle.x = new_x

        scores.append(f(gbest))

        print("Best fitness: ", f(gbest))
        curr_epoch += 1
    return gbest, f(gbest), scores

def rastrigin(x, A=10):
    n = len(x)
    sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return A * n + sum_term

def beale(x):
    x0, x1 = x[0], x[1]
    return (1.5 - x0 + x0*x1)**2 + (2.25 - x0 + x0*x1**2)**2 + (2.625 - x0 + x0*x1**3)**2

def paraboloid(x):
    x0, x1 = x[0], x[1]
    return x0**2 + x1**2


if __name__ == "__main__":
    f_sphere = lambda x: x[0] ** 2 + x[1] ** 2

    global_best, final_eval, scores = particle_swarm_opt(f=f_sphere, p_num=10, n=2, max_epochs=1000000, bounds=(-5.12, 5.12), alph1=1, alph2=1)
    print("Global best: ", global_best, "Evaluation: ", final_eval)
    plt.plot(range(len(scores)), scores, label="Global bests through epochs", color='r')
    plt.ylabel("Best Global Fitness")
    plt.xlabel("Epochs")
    plt.show()

from typing import List
import math
import numpy as np

DIMENSIONS = 2              # Number of dimensions
GLOBAL_BEST = 0             # Global Best of Cost function
B_LO = -5                   # Upper boundary of search space
B_HI = 5                    # Upper boundary of search space

POPULATION = 20             # Number of particles in the swarm
V_MAX = 0.1                 # Maximum velocity value
PERSONAL_C = 2.0            # Personal coefficient factor
SOCIAL_C = 2.0              # Social coefficient factor
CONVERGENCE = 0.001         # Convergence value
MAX_ITER = 100              # Maximum number of iterrations

class Particle():
    def __init__(self, x, y, z, velocity):
        self.position = [x, y]
        self.z = z
        self.velocity = velocity
        self.best_position = [x, y]

class Swarm():
    def __init__(self, population, v_max, cost_function):
        self.particles: List[Particle] = []             # List of particles in the swarm
        self.best_position = None            # Best particle of the swarm
        self.best_z = math.inf      # Best particle of the swarm

        for _ in range(population):
            x = np.random.uniform(B_LO, B_HI)
            y = np.random.uniform(B_LO, B_HI)
            z = cost_function(x, y)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if self.best_position != None and particle.z < self.best_z:
                self.best_position = particle.position.copy()
                self.best_z = particle.z
            else:
                self.best_position = particle.position.copy()
                self.best_z = particle.z

def pso(cost_function):

    swarm = Swarm(POPULATION, V_MAX, cost_function)

    inertia_weight = 0.5 + (np.random.rand()/2)
    
    current_iteration = 0

    while current_iteration < MAX_ITER:

        for particle in swarm.particles:

            for i in range(0, DIMENSIONS):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                
                personal_coefficient = PERSONAL_C * r1 * (particle.best_position[i] - particle.position[i])
                social_coefficient = SOCIAL_C * r2 * (swarm.best_position[i] - particle.position[i])
                new_velocity = inertia_weight * particle.velocity[i] + personal_coefficient + social_coefficient

                if new_velocity > V_MAX:
                    particle.velocity[i] = V_MAX
                elif new_velocity < -V_MAX:
                    particle.velocity[i] = -V_MAX
                else:
                    particle.velocity[i] = new_velocity

            particle.position += particle.velocity
            particle.z = cost_function(particle.position[0], particle.position[1])

            if particle.z < cost_function(particle.best_position[0], particle.best_position[1]):
                particle.best_position = particle.position.copy()

                if particle.z < swarm.best_z:
                    swarm.best_position = particle.position.copy()
                    swarm.best_z = particle.z
                    
            if particle.position[0] > B_HI:
                particle.position[0] = np.random.uniform(B_LO, B_HI)
                particle.z = cost_function(particle.position[0], particle.position[1])
            if particle.position[1] > B_HI:
                particle.position[1] = np.random.uniform(B_LO, B_HI)
                particle.z = cost_function(particle.position[0], particle.position[1])
            if particle.position[0] < B_LO:
                particle.position[0] = np.random.uniform(B_LO, B_HI)
                particle.z = cost_function(particle.position[0], particle.position[1])
            if particle.position[1] < B_LO:
                particle.position[1] = np.random.uniform(B_LO, B_HI)
                particle.z = cost_function(particle.position[0], particle.position[1])

        if abs(swarm.best_z - GLOBAL_BEST) < CONVERGENCE:
            print("The swarm has met convergence criteria after " + str(current_iteration) + " iterrations.")
            break
        current_iteration += 1
    return swarm.best_position, swarm.best_z, current_iteration

def ackley(x, y, a=20, b=0.2, c=2*math.pi):
    arg1 = np.exp((-b * np.sqrt(0.5 * (x ** 2 + y ** 2))))
    arg2 = np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return -1 * a * arg1 - arg2 + a + np.exp(1)

if __name__ == "__main__":
    best_position, best_z, iteration = pso(ackley)
    print("Best position: ", best_position)
    print("Best z: ", best_z)
    
"""! Particle Filter
This module contains the implementation of a particle filter for localization.
"""
import numpy as np
import scipy


class ParticleFilter:
    """
    Particle Filter
    """

    def __init__(self,
                 num_particles: int,
                 occupancy_map: np.ndarray,
                 boundaries: np.ndarray = None,
                 landmarks: np.ndarray = None,
                 ):
        self.num_particles = num_particles
        self.occupancy_map = occupancy_map
        self.boundaries = boundaries
        self.landmarks = landmarks

        # if boundaries are given, add them to the occupancy map
        if boundaries is not None:
            # TODO: add boundaries to the occupancy map
            pass

        # if no landmarks are given, use the boundaries as landmarks
        # if landmarks is None:
        #     self.landmarks = boundaries

        # if landmarks are used, add them to the occupancy map

        # initialize the particles (x, y, theta, weight)
        self.particles = np.zeros((num_particles, 4))

    def start(self):
        """
        Start the particle filter by generating particles
        """
        self.generate_particles(self.num_particles)

    def update(self, motion):
        """
        Update the particle filter
        :param motion: motion model
        :return: estimate of the state
        """
        # predict the next state of the particles
        self.predict_particles(motion)
        # update weights of the particles
        self.update_weights()
        # self.update_particles(motion)
        # self.resample_particles()
        # return self.compute_estimate()

    def calculate_occupancy_map(self):
        """
        Calculate the occupancy map from the given boundaries and landmarks
        """
        pass
        # this strategy is not efficient, would require a quad nested loop very slow
        # TODO: implement a more efficient way to calculate the occupancy map
        # maybe https://www.youtube.com/watch?v=NbSee-XM7WA
        # for i in range(len(self.occupancy_map)):
        #     for j in range(len(self.occupancy_map[0])):
        #         for k in range(4):
        #             if self.cast_ray()

    def generate_particles(self, num_particles):
        """
        Generate random particles in the grid
        :param num_particles: number of particles to generate
        :return: None
        """
        for i in range(num_particles):
            self.particles[i] = self.random_free_place()

    def predict_particles(self, motion, noise=0):
        """
        Predict the next state of the particles
        :param motion: motion model
        :param noise: noise to add to the motion model
        """
        for i in range(self.num_particles):
            self.particles[i] = self.predict_particle(self.particles[i], motion, noise)

    def predict_particle(self, particle, motion, noise=0):
        """
        Move the particle based on the motion model
        :param particle: particle to move
        :param motion: motion model to use
        :param noise: noise to add to the motion model
        :return: moved particle
        """
        # TODO: add noise to the motion model
        # position of the particle
        x, y, theta, weight = particle
        # Motion of the particle
        linear_movement, angular_velocity = motion
        # Update the position of the particle
        x += linear_movement * np.cos(theta)
        y += linear_movement * np.sin(theta)
        # Update the heading of the particle
        theta += angular_velocity
        # TODO: check if the particle should be recreated in a random place
        # Check if the particle is in the grid
        if not self.is_free(x, y):
            # If not, get a new random place
            x, y, theta, weight = self.random_free_place()
        # Return the new particle
        return x, y, theta, weight

    def update_weights(self, measurement=None):
        """
        Update the weights of the particles
        """
        for i in range(self.num_particles):
            self.particles[i] = self.update_weight(self.particles[i], measurement)

    def update_weight(self, particle, measurement):
        """
        Update the weight of a particle
        :param particle: particle to update
        :param measurement: measurement to use
        :return: updated particle
        """
        # if measurement is None, each particle is equally likely to be the true state
        if measurement is None:
            # TODO: degrade the weights of the particles over time if no measurement is given
            raise NotImplementedError()

    @staticmethod
    def cast_ray(wall, particle):
        """
        Cast a ray from the particle to the wall.
        :param wall: wall to cast the ray to
        :param particle: particle to cast the ray from
        :return: intersection point if found, None otherwise
        """
        # tutorial: https://www.youtube.com/watch?v=TOEi6T2mtHo
        # TODO: convert to matrix operations
        x1, y1, x2, y2 = wall
        x3, y3, theta = particle

        x4 = x3 + np.cos(theta)
        y4 = y3 + np.sin(theta)

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and u > 0:
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return None

    def update_particles(self, motion):
        raise NotImplementedError()

    def resample_particles(self):
        raise NotImplementedError()

    def compute_estimate(self):
        raise NotImplementedError()

    def random_free_place(self):
        """
        Get a random free place in the grid
        :return: x, y, theta
        """
        # TODO: add a check for the number of tries, to avoid infinite loops
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y, np.random.uniform(0, 2 * np.pi), -1

    def random_place(self):
        """
        Get a random x,y coordinate in the grid
        :return: x, y
        """
        x = np.random.uniform(0, len(self.occupancy_map))
        y = np.random.uniform(0, len(self.occupancy_map[0]))
        return x, y

    def is_free(self, x, y):
        """
        Check if the given x,y coordinate is free
        :param x: x coordinate
        :param y: y coordinate
        :return: True if free, False otherwise
        """
        if not self.is_in(x, y):
            return False
        yy = int(y)
        xx = int(x)
        return self.occupancy_map[xx, yy] == 0

    def is_in(self, x, y):
        """
        Check if the given x,y coordinate is in the grid
        :param x: x coordinate
        :param y: y coordinate
        :return: True if in, False otherwise
        """
        if x < 0 or y < 0 or x > len(self.occupancy_map) or y > len(self.occupancy_map[0]):
            return False
        return True


if __name__ == "__main__":
    m = np.zeros((100, 100))
    # add walls to the edges
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    print(m)
    pf = ParticleFilter(1000, m)

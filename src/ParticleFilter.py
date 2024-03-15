"""! Particle Filter
This module contains the implementation of a particle filter for localization.
"""
import numpy as np


class ParticleFilter:
    """
    Particle Filter
    """

    def __init__(self,
                 occupancy_map: np.ndarray = None,
                 num_particles: int = 1000,
                 boundaries: np.ndarray = None,
                 landmarks: np.ndarray = None,
                 ):
        self.num_particles = num_particles

        # 1. occupancy_map is given, boundaries and landmarks are None
        if occupancy_map is not None and boundaries is None and landmarks is None:
            # use the given occupancy map, and calculate boundaries based on the map
            self.occupancy_map = occupancy_map
            self.landmarks = None
            raise NotImplementedError("Calculate boundaries based on the occupancy map")
        # 2. occupancy_map is given, boundaries are given, landmarks are None
        elif occupancy_map is not None and boundaries is not None and landmarks is None:
            # use the given occupancy map and boundaries
            self.occupancy_map = occupancy_map
            self.boundaries = boundaries
            self.landmarks = None
        # 3. occupancy_map is given, boundaries are given, landmarks are given
        elif occupancy_map is not None and boundaries is not None and landmarks is not None:
            self.occupancy_map = occupancy_map
            self.boundaries = boundaries
            self.landmarks = landmarks
        # 4. occupancy_map is None, boundaries are given, landmarks are None
        elif occupancy_map is None and boundaries is not None and landmarks is None:
            # calculate the occupancy map based on the given boundaries
            raise NotImplementedError("Calculate occupancy map based on the given boundaries")
        # 5. occupancy_map is None, boundaries are given, landmarks are given
        elif occupancy_map is None and boundaries is not None and landmarks is not None:
            # calculate the occupancy map based on the given boundaries and landmarks
            raise NotImplementedError("Calculate occupancy map based on the given boundaries and landmarks")
        # 6. occupancy_map is None, boundaries are None, landmarks are given
        elif occupancy_map is None and boundaries is None and landmarks is not None:
            # calculate the occupancy map based on the given landmarks
            raise NotImplementedError("Calculate occupancy map based on the given landmarks")
        # 7. occupancy_map is None, boundaries are None, landmarks are None
        else:
            # create default occupancy map, boundaries and landmarks
            raise NotImplementedError("Create default occupancy map, boundaries and landmarks")

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

    def update(self, motion, measurements=None):
        """
        Update the particle filter
        :param measurements: np.ndarray of measurements
        :param motion: motion model
        :return: estimate of the state
        """
        # predict the next state of the particles
        self.predict_particles(motion)
        # update weights of the particles
        self.update_weights(measurements)
        # resample the particles
        self.resample_particles()
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
            try:
                self.particles[i] = self.update_weight(self.particles[i], measurement)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise e

    def update_weight(self, particle, measurement):
        """
        Update the weight of a particle.
        :param particle: particle to update
        :param measurement: measurement to use
        :return: updated particle
        """
        # if measurement is None, each particle is equally likely to be the true state
        if measurement is None:
            # TODO: degrade the weights of the particles over time if no measurement is given
            return particle
        closest_wall = None
        for boundary in self.boundaries:
            intersection = self.cast_ray(particle, boundary)
            if intersection is not None:
                x, y = intersection
                # distance to the wall
                d = np.sqrt((particle[0] - x) ** 2 + (particle[1] - y) ** 2)
                if closest_wall is None or d < closest_wall[0]:
                    closest_wall = d, intersection
        if closest_wall is None:
            weight = 0
        else:
            error = np.abs((closest_wall[0] - measurement) / measurement) * 100
            if error > 100:
                weight = 0
            else:
                weight = 100 - error
        particle[3] = weight
        return particle

    @staticmethod
    def cast_ray(particle, wall, d_angle=0):
        """
        Cast a ray from the particle to the wall.
        :param wall: wall to cast the ray to
        :param particle: particle to cast the ray from (
        :param d_angle: delta angle to add to the particle's heading in radians
        :return: intersection point if found, None otherwise
        """
        # tutorial: https://www.youtube.com/watch?v=TOEi6T2mtHo
        # TODO: convert to matrix operations
        x1, y1, x2, y2 = wall
        x3, y3, theta, _ = particle

        x4 = x3 + np.cos(theta + d_angle)
        y4 = y3 + np.sin(theta + d_angle)

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and u > 0:
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return None

    def resample_particles(self):
        """
        Resample the particles, discarding the ones with low weights and duplicating the ones with high weights
        :return:
        """
        self.particles = self.particles[self.particles[:, 3].argsort()]
        for i in range(len(self.particles)):
            if self.particles[i][3] == 0:
                # TODO: replace arbitrary value with dynamic value
                self.particles[i] = self.particles[np.random.randint(0, 10)]

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
                rotation = np.random.uniform(0, 2 * np.pi)
                return x, y, rotation, -1

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
    pf = ParticleFilter(m, 1000)

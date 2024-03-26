"""! Particle Filter
This module contains the implementation of a particle filter for localization.
"""
import math
import time

import numpy as np
import multiprocessing


class ParticleFilter:
    """
    Particle Filter
    """

    def __init__(self,
                 occupancy_map: np.ndarray = None,
                 num_particles: int = 1000,
                 boundaries: np.ndarray = None,
                 landmarks: np.ndarray = None,
                 sensor_angles: np.ndarray = None,
                 use_multiprocessing: bool = False,
                 ):
        self.num_particles = num_particles

        # 1. occupancy_map is given, boundaries and landmarks are None
        if occupancy_map is not None and boundaries is None and landmarks is None:
            # use the given occupancy map, and calculate boundaries based on the map
            self.occupancy_map = occupancy_map
            self.landmarks = np.array([])
            raise NotImplementedError("Calculate boundaries based on the occupancy map")
        # 2. occupancy_map is given, boundaries are given, landmarks are None
        elif occupancy_map is not None and boundaries is not None and landmarks is None:
            # use the given occupancy map and boundaries
            self.occupancy_map = occupancy_map
            self.boundaries = boundaries
            self.landmarks = np.array([])
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

        if sensor_angles is None:
            self.sensor_angles = [0]
        else:
            self.sensor_angles = sensor_angles

        # initialize the particles (x, y, theta, weight)
        self.particles_dtype = np.dtype(
            [('x', np.float64), ('y', np.float64), ('theta', np.float64), ('weight', np.float64)])
        # particles = np.zeros((num_particles, 4))
        # self.particles = np.array(particles, dtype=self.particles_dtype)
        self.particles = np.zeros((num_particles, 4))

        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing:
            try:
                import multiprocessing
            except ImportError:
                raise ImportError("Multiprocessing is not supported on this system")

    def start(self, starting_pose=None):
        """
        Start the particle filter by generating particles
        """
        print("Starting the particle filter, starting pose:", starting_pose)
        self.generate_particles(self.num_particles, starting_pose=starting_pose)

    def update(self, motion, sensor_measurements=None, sensor_angles=None, landmark_measurements=None):
        """
        Update the particle filter
        :param motion: motion model
        :param sensor_measurements: np.ndarray of measurements
        :param sensor_angles: np.ndarray angles to use for the ray cast
        :param landmark_measurements: np.ndarray of landmarks
        :return: estimate of the state
        """
        start_predict_time = time.time_ns()
        # predict the next state of the particles
        self.predict_particles(motion)
        end_predict_time = time.time_ns()
        if sensor_measurements is None and landmark_measurements is None:
            print("No measurements or landmarks given, returning the estimate of the state")
            # if no measurements are given, return the estimate of the state
            return self.compute_estimate()
        start_update_time = time.time_ns()
        # update weights of the particles
        self.update_weights(sensor_measurements, sensor_angles, landmark_measurements)
        end_update_time = time.time_ns()
        start_resample_time = time.time_ns()
        # resample the particles
        self.resample_particles()
        end_resample_time = time.time_ns()
        start_estimate_time = time.time_ns()
        # return the estimate of the state
        # return self.compute_estimate()
        estimate = self.compute_estimate()
        end_estimate_time = time.time_ns()
        print(f"predict time: {(end_predict_time - start_predict_time) / 1e6} ms, "
              f"update time: {(end_update_time - start_update_time) / 1e6} ms, "
              f"resample time: {(end_resample_time - start_resample_time) / 1e6} ms, "
              f"estimate time: {(end_estimate_time - start_estimate_time) / 1e6} ms")
        return estimate

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

    def generate_particles(self, num_particles, starting_pose=None):
        """
        Generate random particles in the grid
        :param num_particles: number of particles to generate
        :return: None
        """
        if starting_pose is None:
            for i in range(num_particles):
                self.particles[i] = self.random_free_place()
        else:
            for i in range(1, num_particles):
                self.particles[i] = (starting_pose[0], starting_pose[1], starting_pose[2], -1)

    def predict_particles(self, motion, linear_noise=0, angular_noise=0.01):
        """
        Predict the next state of the particles
        :param motion: motion model
        :param linear_noise: noise to add to the motion model linear velocity
        :param angular_noise: noise to add to the motion model angular velocity
        """
        for i in range(self.num_particles):
            if i == 0:
                self.particles[i] = self.predict_particle(self.particles[i], motion,
                                                          linear_noise=linear_noise,
                                                          angular_noise=angular_noise,
                                                          # debug=True
                                                          )
            else:
                self.particles[i] = self.predict_particle(self.particles[i], motion,
                                                          linear_noise=linear_noise,
                                                          angular_noise=angular_noise)

    def predict_particle(self, particle, motion, linear_noise=0.1, angular_noise=0.01, debug=False):
        """
        Move the particle based on the motion model
        :param particle: particle to move
        :param motion: motion model to use (linear, angular)
        :param linear_noise: noise to add to the motion model linear velocity
        :param angular_noise: noise to add to the motion model angular velocity
        :return: moved particle
        """
        # position of the particle
        x, y, theta, weight = particle
        # Motion of the particle
        linear, angular = motion
        # add noise to motion
        linear = np.random.normal(linear, linear_noise)
        angular = np.random.normal(angular, angular_noise)
        # Update the position of the particle
        x += linear * np.cos(theta)
        y += linear * np.sin(theta)
        # Update the heading of the particle
        theta += angular
        # i = 0
        # while True:
        #     if theta < 0:
        #         theta += 2 * np.pi
        #     elif theta > 2 * np.pi:
        #         theta -= 2 * np.pi
        #     else:
        #         break
        #     i += 1
        #     if i > 10:
        #         raise ValueError("could not find a valid theta for new particle")

        # Check if the particle is in the grid
        if not self.is_free(x, y):
            # If not, get a new random place
            x, y, theta, weight = self.random_free_place()

        if debug:
            print(f"particle: {particle} -> {x, y, theta, weight}\nmotion: {motion} -> {linear, angular}")
        # Return the new particle
        return x, y, theta, weight

    def update_weights(self, measurements=None, sensor_angles=None, landmarks=None):
        """
        Update the weights of the particles
        """
        if measurements is None and landmarks is None:
            raise ValueError("No measurements or landmarks given")
        elif measurements is not None and sensor_angles is not None:
            if len(measurements) != len(sensor_angles):
                raise ValueError(f"Number of measurements should be equal to the number of sensor angles len "
                                 f"measurements:{len(measurements)} len sensor angles:{len(sensor_angles)}\n"
                                 f"{measurements}\n{sensor_angles}")
            if self.use_multiprocessing is True:
                print("Using multiprocessing")
                cpu_cores = multiprocessing.cpu_count()
                # TODO: add variable number of processes
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(self.update_weight,
                                           [(self.particles[i], measurements, 0.1, sensor_angles) for i in
                                            range(self.num_particles)], chunksize=100)
                    # results = pool.starmap(self.update_weight, self.particles, chunksize=100)
                    self.particles = np.array(results)
            else:
                for i in range(self.num_particles):
                    if i == 0:
                        self.particles[i] = self.update_weight(self.particles[i], measurements,
                                                               raycast_angles=sensor_angles, debug=True)
                    else:
                        self.particles[i] = self.update_weight(self.particles[i], measurements,
                                                               raycast_angles=sensor_angles)
        elif measurements is not None and sensor_angles is None:
            raise ValueError("Sensor angles are not given")
        else:
            raise NotImplementedError("Landmark based localization is not implemented")

    def update_weight(self, particle, measurements, noise=0.01, raycast_angles=None, debug=False):
        """
        Update the weight of a particle.
        :param particle: particle to update
        :param measurements: measurements to use
        :param noise: noise to add to the measurements
        :param raycast_angles: angles to use for the ray cast
        :return: updated particle
        """

        if raycast_angles is None:
            raycast_angles = [0]

        if len(measurements) != len(raycast_angles):
            raise ValueError("Number of measurements should be equal to the number of measurement models")

        running_weight = 0
        measured = []
        for i in range(len(measurements)):
            measurement = measurements[i]
            # if measurement is None, each particle is equally likely to be the true state
            if measurement is None:
                # TODO: degrade the weights of the particles over time if no measurement is given
                continue
            closest_wall = None
            for boundary in self.boundaries:
                intersection = self.cast_ray(particle, boundary, d_angle=raycast_angles[i])
                if intersection is not None:
                    x, y = intersection
                    if noise > 0:
                        x += np.random.normal(0, noise)
                        y += np.random.normal(0, noise)
                    # distance to the wall
                    d = np.sqrt((particle[0] - x) ** 2 + (particle[1] - y) ** 2)
                    if closest_wall is None or d < closest_wall[0]:
                        closest_wall = d, intersection
            if closest_wall is None:
                weight = 0
                measured.append(float("inf"))
            else:
                measured.append(closest_wall[0])
                error = np.abs((closest_wall[0] - measurement) / measurement) * 100
                if error > 100:
                    weight = 0
                else:
                    weight = 100 - error
            running_weight += weight
        weight = running_weight / len(measurements)
        prev_weight = particle[3]
        if prev_weight == -1:
            particle[3] = weight
        else:
            # update the weight using a moving average
            particle[3] = (prev_weight + weight) / 2

        if debug:
            print(
                f"particle: {particle} -> {particle[3]}\nmeasurements: {measurements} -> {measured}\nweight: {prev_weight} -> {weight}\n")

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

    def calculate_obstacle_occupancy(self, obsticle):
        """
        Given an obstacle, calculate the occupancy of the obstacle and update the occupancy map
        :param obsticle: obstacle to calculate occupancy for
        :return: None
        """
        # tutorial: https://www.youtube.com/watch?v=NbSee-XM7WA
        x1, y1, x2, y2 = obsticle

        v_ray_start = np.array([x1, y1])
        v_ray_dist = np.array([x2 - x1, y2 - y1])
        max_distance = math.dist([x1, y1], [x2, y2])
        print(max_distance)
        if v_ray_dist[0] == 0 and v_ray_dist[1] == 0:
            raise ValueError("Invalid obstacle, start and end are the same")
        # elif v_ray_dist[0] == 0:
        #     # TODO: the ray is vertical
        #     print("vertical ray, skipping...")
        #     return
        # elif v_ray_dist[1] == 0:
        #     # TODO: the ray is horizontal
        #     print("horizontal ray, skipping...")
        #     return
        v_ray_dir = v_ray_dist / np.sqrt(np.sum(v_ray_dist ** 2))

        ray_unit_step_size = (
            np.sqrt(1 + ((v_ray_dir[1] / v_ray_dir[0]) ** 2)), np.sqrt(1 + ((v_ray_dir[0] / v_ray_dir[1]) ** 2)))

        if ray_unit_step_size[0] == np.inf:
            ray_unit_step_size = (0, ray_unit_step_size[1])
        if ray_unit_step_size[1] == np.inf:
            ray_unit_step_size = (ray_unit_step_size[0], 0)
        if v_ray_dir[0] == 0 and v_ray_dir[1] == 0:
            raise ValueError("Invalid obstacle, start and end are the same")

        v_map_check = np.array([int(v_ray_start[0]), int(v_ray_start[1])])
        v_ray_len_1d = np.array([0, 0])
        v_step = np.array([0, 0])

        print(v_ray_start, v_ray_dist, max_distance, v_ray_dir, ray_unit_step_size, v_map_check, v_ray_len_1d, v_step)

        if v_ray_dir[0] < 0:
            v_step[0] = -1
            v_ray_len_1d[0] = (v_ray_start[0] - v_map_check[0]) * ray_unit_step_size[0]
        else:
            v_step[0] = 1
            v_ray_len_1d[0] = ((v_map_check[0] + 1.0) - v_ray_start[0]) * ray_unit_step_size[0]

        if v_ray_dir[1] < 0:
            v_step[1] = -1
            v_ray_len_1d[1] = (v_ray_start[1] - v_map_check[1]) * ray_unit_step_size[1]
        else:
            v_step[1] = 1
            v_ray_len_1d[1] = ((v_map_check[1] + 1.0) - v_ray_start[1]) * ray_unit_step_size[1]

        # max_distance = 10.0
        f_distance = 0.0
        while f_distance < max_distance:
            if v_ray_len_1d[0] < v_ray_len_1d[1]:
                v_map_check[0] += v_step[0]
                v_ray_len_1d[0] += ray_unit_step_size[0]
            else:
                v_map_check[1] += v_step[1]
                v_ray_len_1d[1] += ray_unit_step_size[1]
            f_distance = np.sqrt(np.sum((v_map_check - v_ray_start) ** 2))

            if (0 <= v_map_check[0] < len(self.occupancy_map) and
                    0 <= v_map_check[1] < len(self.occupancy_map[0])):
                if self.occupancy_map[v_map_check[0], v_map_check[1]] == 0:
                    self.occupancy_map[v_map_check[0], v_map_check[1]] = 1
            else:
                break

    def resample_particles(self):
        """
        Resample the particles, discarding the ones with low weights and duplicating the ones with high weights
        :return:
        """
        print(f"pre resample: {self.particles[0:5, 3]}")
        # sort the particles based on the weights
        self.particles = self.particles[self.particles[:, 3].argsort()]
        print(f"sorted: {self.particles[0:5, 3]}")
        # get number of particles that are above a certain threshold
        n = len(self.particles[self.particles[:, 3] > 50])
        if n == 0:
            print("All particles have low weights, resampling all particles")
            for i in range(len(self.particles)):
                self.particles[i] = self.random_free_place()
            return
        for i in range(len(self.particles)):
            if self.particles[i][3] < 10:
                # self.particles[i] = self.particles[np.random.choice(n)]
                self.particles[i] = self.random_near_particle(self.particles[np.random.randint(n)])

    def compute_estimate(self):
        """
        Compute the estimate of the state
        :return: estimate of the state (x, y, theta)
        """
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        theta = np.mean(self.particles[:, 2])
        return x, y, theta

    def random_free_place(self):
        """
        Get a random free place in the grid
        :return: x, y, theta
        """
        # TODO: add a check for the number of tries, to avoid infinite loops
        i = 0
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                rotation = np.random.uniform(0, 2 * np.pi)
                return x, y, rotation, -1
            i += 1
            if i > 100000:
                raise ValueError("Could not find a free place in the grid")

    def random_near_particle(self, particle, position_noise=0, angular_noise=0):
        """
        Get a random place near the given particle.
        :param particle: particle to get a random place near
        :param position_noise: noise to add to the random place
        :param angular_noise: noise to add to the random angle
        :return: x, y
        """
        x, y, theta, weight = particle
        if position_noise != 0:
            i = 0
            while True:
                new_x = np.random.normal(x, position_noise)
                new_y = np.random.normal(y, position_noise)
                if self.is_free(new_x, new_y):
                    break
                i += 1
                if i > 100000:
                    print(new_x, new_y)
                    raise ValueError("Could not find a free place near the particle")
        else:
            new_x, new_y = x, y

        if angular_noise == 0:
            return new_x, new_y, theta, weight

        new_theta = np.random.normal(theta, angular_noise)
        i = 0
        while True:
            if new_theta < 0:
                new_theta += 2 * np.pi
            elif new_theta > 2 * np.pi:
                new_theta -= 2 * np.pi
            else:
                break
            i += 1
            if i > 10:
                print(f"could not find a valid theta for new particle, randomizing")
                new_theta = np.random.uniform(0, 2 * np.pi)

        return new_x, new_y, new_theta, weight

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

    def add_landmarks(self, landmarks: np.ndarray):
        """
        Add landmarks to the map
        :param landmarks: list of landmarks, landmarks have the format (x, y, theta, id)
        """
        np.append(self.landmarks, landmarks)


if __name__ == "__main__":
    m = np.zeros((100, 100))
    # add walls to the edges
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    print(m)
    pf = ParticleFilter(m, 1000)

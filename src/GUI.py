"""! GUI
This module contains the GUI for the particle filter. The GUI is implemented using the tkinter library.
"""
import tkinter as tk
import numpy as np
from ParticleFilter import ParticleFilter
import time


class ParticleFilterGUI:
    """
    GUI for the particle filter
    """
    estimated_pose_size = 10

    def __init__(self,
                 particle_filter: ParticleFilter,  # particle_filter is an instance of ParticleFilter
                 box_size: int = 10,  # size of a box in the grid in pixels
                 pose_size: int = 10,  # size of the pose in pixels
                 particle_size: int = 4,  # size of the particle in pixels
                 landmark_size: int = 10,  # size of the landmark in pixels
                 real_pose: tuple = None,  # real pose of the robot this is used for testing only
                 ):

        # TEMP: for testing only

        self.real_pose = real_pose
        self.estimated_pose = None
        self.last_key_pressed = None
        self.move_distance = 5
        self.turn_angle = np.pi / 4

        # END TEMP

        self.box_size = box_size
        self.pose_size = pose_size
        self.particle_size = particle_size
        self.landmark_size = landmark_size

        self.particle_filter = particle_filter

        self.root = tk.Tk()
        self.root.title("Particle Filter GUI")

        # Create a canvas
        self.canvas = tk.Canvas(self.root, width=len(pf.occupancy_map) * self.box_size,
                                height=len(pf.occupancy_map[0]) * self.box_size)

        # Pack the canvas into the window
        self.canvas.pack(side="top", fill="both", expand=True)
        self.initialize_gui()
        self.root.after(0, self.update)
        self.root.bind("<Key>", self.key_pressed)
        self.root.mainloop()

    def initialize_gui(self):
        """
        Initialize the GUI so that it shows the grid and the particles at the start. This is necessary because the
        update method will only update the canvas if there is a change in the state of the particle filter,
        so we need to draw the initial state of the particle filter.
        :return: None
        """
        self.canvas.delete("all")  # probably not necessary but good practice
        self.draw_grid()
        self.draw_particles(self.particle_filter.particles)
        self.draw_real_pose(self.real_pose)
        # self.draw_estimated_poses()
        self.draw_landmarks(self.particle_filter.landmarks)
        self.draw_boundaries(self.particle_filter.boundaries)

    def update(self):
        """
        Update the GUI
        :return: None
        """

        start_time = time.time_ns()

        # updates
        update_all = False
        update_obstacles = False
        update_particles = True
        updated_real_pose = True
        update_estimated_poses = True

        # get user input
        # requested_motion = self.get_user_input()
        requested_motion = self.random_motion()
        # update_particles = requested_motion != (0, 0)

        # update the real poses of the robots
        if requested_motion != (0, 0):
            distance, angle = requested_motion
            x, y, theta, _ = self.real_pose
            x += np.cos(theta) * distance
            y += np.sin(theta) * distance
            theta += angle
            self.real_pose = (x, y, theta, 1)

        # angles = [np.pi / 4, 0, 11 * np.pi / 6]
        angles = np.linspace(0, 2 * np.pi, 11)[:-1]
        distances = []
        intersections = []
        for angle in angles:
            closest_wall = None
            for boundary in self.particle_filter.boundaries:
                intersection = self.particle_filter.cast_ray(self.real_pose, boundary, angle)
                if intersection is not None:
                    x, y = intersection
                    # distance to the wall
                    d = np.sqrt((self.real_pose[0] - x) ** 2 + (self.real_pose[1] - y) ** 2)
                    if closest_wall is None or d < closest_wall[0]:
                        closest_wall = d, intersection
            if closest_wall is not None:
                distances.append(closest_wall[0])
                intersections.append(closest_wall[1])
            else:
                distances.append(float("inf"))
                intersections.append((self.real_pose[0], self.real_pose[1]))

        # print(len(distances), len(angles), len(intersections), distances)
        # update the particle filter
        pf_start_time = time.time_ns()
        self.estimated_pose = self.particle_filter.update(requested_motion, sensor_measurements=distances,
                                                          sensor_angles=angles)
        pf_end_time = time.time_ns()

        gui_start_time = time.time_ns()
        if update_all or update_obstacles and update_particles and updated_real_pose and update_estimated_poses:
            self.canvas.delete("all")
            self.draw_grid()
            self.draw_boundaries(self.particle_filter.boundaries)
            self.draw_landmarks(self.particle_filter.landmarks)
            self.draw_particles(self.particle_filter.particles)
            self.draw_real_pose(self.real_pose, intersections)
            self.draw_estimated_poses(self.estimated_pose)
        elif update_particles and updated_real_pose and update_estimated_poses:
            self.canvas.delete("particle")
            self.canvas.delete("real_pose")
            self.canvas.delete("estimated_pose")
            self.draw_particles(self.particle_filter.particles)
            self.draw_real_pose(self.real_pose, intersections)
            self.draw_estimated_poses(self.estimated_pose)
        elif update_particles or updated_real_pose:
            self.canvas.delete("particle")
            self.canvas.delete("real_pose")
            self.draw_particles(self.particle_filter.particles)
            self.draw_real_pose(self.real_pose, intersections)
        elif update_obstacles:
            self.canvas.delete("obstacle")
            self.canvas.delete("free")
            self.draw_grid()
        elif update_estimated_poses:
            self.canvas.delete("estimated_pose")
            self.draw_estimated_poses(self.estimated_pose)
        else:
            # print("No updates")
            pass
        gui_end_time = time.time_ns()
        end_time = gui_end_time
        print(
            f"update time (ms): {(end_time - start_time) / 1e+6}, pf time (ms): {(pf_end_time - pf_start_time) / 1e+6}"
            f", gui time (ms): {(gui_end_time - gui_start_time) / 1e+6}")

        # self.particle_filter.update(motion=(np.random.normal(0, 5), np.random.normal(0, 2 * np.pi)))
        self.root.after(1, self.update)

    def random_motion(self):
        """
        Generate a random legal motion for the robot
        :return: (distance, angle) where distance is the distance to move and angle is the angle to turn
        """
        i = 0
        while True:
            distance, angle = np.random.normal(0, 5), np.random.normal(np.pi / 4, np.pi / 4)
            new_x = self.real_pose[0] + np.cos(self.real_pose[2]) * distance
            new_y = self.real_pose[1] + np.sin(self.real_pose[2]) * distance
            if self.particle_filter.is_free(new_x, new_y):
                return distance, angle
            i += 1
            if i > 100:
                raise Exception("No legal motion found")

    def key_pressed(self, event):
        """
        Handle key pressed event
        :param event:
        :return:
        """
        self.last_key_pressed = event.char

    def get_user_input(self):
        """
        Get user input for the motion model
        :return: requested motion, (linear, angular)
        """
        move_distance = 0
        turn_angle = 0
        if self.last_key_pressed is not None:
            if self.last_key_pressed == "w":
                move_distance = self.move_distance
            elif self.last_key_pressed == "s":
                move_distance = -self.move_distance
            elif self.last_key_pressed == "a":
                turn_angle = -self.turn_angle
            elif self.last_key_pressed == "d":
                turn_angle = self.turn_angle
            self.last_key_pressed = None
        return move_distance, turn_angle

    def draw_grid(self):
        """
        Draw the grid - white for free space and black for obstacles
        :return: None
        """
        for i in range(len(self.particle_filter.occupancy_map)):
            for j in range(len(self.particle_filter.occupancy_map[0])):
                if self.particle_filter.occupancy_map[i, j] > 0:
                    self.draw_box(i, j, color="black", tag="obstacle")
                else:
                    self.draw_box(i, j, color="light grey", tag="free")

    def draw_box(self, x, y, color="black", tag=None):
        """
        Draw a box on the canvas
        :param x: The x coordinate
        :param y: The y coordinate
        :param color: The color of the box
        :param tag: The tag of the box
        :return: None
        """
        if tag is not None:
            self.canvas.create_rectangle(x * self.box_size,
                                         y * self.box_size,
                                         (x + 1) * self.box_size,
                                         (y + 1) * self.box_size,
                                         fill=color, tags=tag)
        else:
            self.canvas.create_rectangle(x * self.box_size,
                                         y * self.box_size,
                                         (x + 1) * self.box_size,
                                         (y + 1) * self.box_size,
                                         fill=color)

    def draw_real_pose(self, pose, rays=None):
        """
        Draw the poses on the canvas, each pose represent the estimated pose of a robot
        :param pose: List of poses to draw, each pose is a tuple (x, y, theta)
        :param rays: List of rays to draw, each ray is a tuple (x, y)
        :return:
        """
        x, y, theta, _ = pose

        if rays is not None:
            for ray in rays:
                self.canvas.create_line(x * self.box_size,
                                        y * self.box_size,
                                        ray[0] * self.box_size,
                                        ray[1] * self.box_size,
                                        fill="teal", tags="real_pose", width=2)

        self.canvas.create_oval((x * self.box_size) - (self.pose_size / 2),
                                (y * self.box_size) - (self.pose_size / 2),
                                (x * self.box_size) + (self.pose_size / 2),
                                (y * self.box_size) + (self.pose_size / 2),
                                fill="blue", outline="blue", tags="real_pose")
        # Draw a line to show the orientation of the pose
        self.canvas.create_line(x * self.box_size,
                                y * self.box_size,
                                (x * self.box_size) + (np.cos(theta) * self.pose_size * 2),
                                (y * self.box_size) + (np.sin(theta) * self.pose_size * 2),
                                fill="orange", tags="real_pose")

    def draw_estimated_poses(self, pose):
        """
        Draw the poses on the canvas, each pose represent the estimated pose of a robot
        :param pose: List of poses to draw, each pose is a tuple (x, y, theta)
        :return:
        """
        # x, y, theta = pose
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        # Draw a blue dot for each pose
        self.canvas.create_oval((x * self.box_size) - (self.estimated_pose_size / 2),
                                (y * self.box_size) - (self.estimated_pose_size / 2),
                                (x * self.box_size) + (self.estimated_pose_size / 2),
                                (y * self.box_size) + (self.estimated_pose_size / 2),
                                fill="green", outline="green", tags="estimated_pose")
        # Draw a line to show the orientation of the pose
        self.canvas.create_line(x * self.box_size,
                                y * self.box_size,
                                (x * self.box_size) + (np.cos(theta) * self.estimated_pose_size * 2),
                                (y * self.box_size) + (np.sin(theta) * self.estimated_pose_size * 2),
                                fill="orange", tags="estimated_pose")

    def draw_particles(self, particles):
        """
        Draw the particles on the canvas
        :return: None
        """
        for particle in particles:
            x, y, theta, prob = particle

            if prob < 0:
                color = "black"
            elif 0 < prob < 25:
                color = "red"
            elif 25 < prob < 50:
                color = "deep pink"
            elif 50 < prob < 75:
                color = "yellow"
            else:
                color = "green"

            # Draw a red dot for each particle
            self.canvas.create_oval((x * self.box_size) - (self.particle_size / 2),
                                    (y * self.box_size) - (self.particle_size / 2),
                                    (x * self.box_size) + (self.particle_size / 2),
                                    (y * self.box_size) + (self.particle_size / 2),
                                    fill=color, outline=color, tags="particle")
            # Draw a line to show the orientation of the particle
            self.canvas.create_line(x * self.box_size,
                                    y * self.box_size,
                                    (x * self.box_size) + (np.cos(theta) * 10),
                                    (y * self.box_size) + (np.sin(theta) * 10),
                                    fill="black", tags="particle")

    def draw_boundaries(self, boundaries):
        """
        Draw the boundaries on the canvas
        :param boundaries: List of boundaries to draw, each boundary is a tuple (x1, y1, x2, y2)
        :return: None
        """
        for boundary in boundaries:
            x1, y1, x2, y2 = boundary
            self.canvas.create_line(x1 * self.box_size,
                                    y1 * self.box_size,
                                    x2 * self.box_size,
                                    y2 * self.box_size,
                                    fill="orange", tags="boundary", width=2)

    def draw_landmarks(self, landmarks):
        """
        Draw the landmarks on the canvas
        :param landmarks: List of landmarks to draw, each landmark is a tuple (x, y)
        :return: None
        """
        for landmark in landmarks:
            x, y, theta = landmark
            # Draw a purple dot for each landmark
            self.canvas.create_oval((x * self.box_size) - (self.landmark_size / 2),
                                    (y * self.box_size) - (self.landmark_size / 2),
                                    (x * self.box_size) + (self.landmark_size / 2),
                                    (y * self.box_size) + (self.landmark_size / 2),
                                    fill="purple", outline="purple", tags="landmark")

            # Draw a line to show the orientation of the landmark
            self.canvas.create_line(x * self.box_size,
                                    y * self.box_size,
                                    (x * self.box_size) + (np.cos(theta) * self.landmark_size * 2),
                                    (y * self.box_size) + (np.sin(theta) * self.landmark_size * 2),
                                    fill="black", tags="landmark")


if __name__ == "__main__":
    m = np.zeros((100, 100))
    # add walls to the edges
    m[0, :] = 1
    m[-1, :] = 1
    m[:, 0] = 1
    m[:, -1] = 1
    print(m)

    lm = np.array([
        (10, 10, .25 * np.pi),
        (20, 20, .75 * np.pi),
        (30, 30, 1.25 * np.pi)
    ])

    bounds = np.array([(0.5, 0.5, 0.5, 99.5),
                       (0.5, 0.5, 99.5, 0.5),
                       (99.5, 0.5, 99.5, 99.5),
                       (0.5, 99.5, 99.5, 99.5),
                       (30, 0.5, 30, 30),
                       (0.5, 30, 30, 30),
                       (75, 75, 75, 80),
                       (75, 75, 80, 75),
                       (75, 80, 80, 80),
                       (80, 75, 80, 80),
                       ])


    def generate_random_squares(num_squares):
        """
        Generate random squares
        :param num_squares: number of squares to generate
        :return: list of squares, each square is four lines each line is a tuple (x1, y1, x2, y2)
        """
        lines = []
        for _ in range(num_squares):
            # Randomly choose the bottom-left corner of the square
            x1 = np.random.randint(0, 90)  # Leave space for the square
            y1 = np.random.randint(0, 90)  # Leave space for the square

            # Assume the square's side length is 10
            x2 = x1 + 10
            y2 = y1 + 10

            # Add the square's coordinates to the list
            # squares.append((x1, y1, x2, y2))
            lines.append([(x1, y1, x2, y1),  # Bottom line
                          (x2, y1, x2, y2),  # Right line
                          (x2, y2, x1, y2),  # Top line
                          (x1, y2, x1, y1)]  # Left line
                         )

        return lines


    boxes = generate_random_squares(5)
    for box in boxes:
        bounds = np.concatenate((bounds, box), axis=0)

    pf = ParticleFilter(m, 1000, boundaries=bounds, use_multiprocessing=True)

    for b in bounds:
        pf.calculate_obstacle_occupancy(b)

    # pf.calculate_obstacle_occupancy(bounds[0])
    # pf.calculate_obstacle_occupancy(bounds[1])
    # pf.calculate_obstacle_occupancy(bounds[6])

    # for b in bounds:
    #     pf.calculate_obstacle_occupancy(b)
    # exit(0)

    pf.add_landmarks(lm)

    possible_angles = np.linspace(0, 2 * np.pi, 8)
    real_x, real_y, _, _ = pf.random_free_place()
    _real_pose = (real_x, real_y, np.random.choice(possible_angles), 1)

    # pf.start(starting_pose=_real_pose)
    pf.start()

    gui = ParticleFilterGUI(pf, real_pose=_real_pose)

    # while True:
    #     # pf.update()
    #     gui.update()

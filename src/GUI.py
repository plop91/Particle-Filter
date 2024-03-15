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

    def __init__(self,
                 particle_filter: ParticleFilter,  # particle_filter is an instance of ParticleFilter
                 box_size: int = 10,  # size of a box in the grid in pixels
                 pose_size: int = 10,  # size of the pose in pixels
                 particle_size: int = 4,  # size of the particle in pixels
                 landmark_size: int = 10,  # size of the landmark in pixels
                 ):

        # TEMP: for testing only

        self.real_pose = (10, 10, np.random.uniform(0, 2 * np.pi), 1)
        self.last_key_pressed = None
        self.move_distance = 5
        self.turn_angle = np.pi / 2

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
        # updates
        update_all = False
        update_obstacles = False
        update_particles = False
        update_poses = False

        # get user input
        requested_motion = self.get_user_input()
        update_poses = requested_motion != (0, 0)
        update_particles = requested_motion != (0, 0)

        # update the real poses of the robots
        if requested_motion != (0, 0):
            distance, angle = requested_motion
            x, y, theta, _ = self.real_pose
            x += np.cos(theta) * distance
            y += np.sin(theta) * distance
            theta += angle
            self.real_pose = (x, y, theta, 1)

        ray_distance = None
        for boundary in self.particle_filter.boundaries:
            intersection = self.particle_filter.cast_ray(self.real_pose, boundary)
            if intersection is not None:
                x, y = intersection
                # distance to the wall
                d = np.sqrt((self.real_pose[0] - x) ** 2 + (self.real_pose[1] - y) ** 2)
                if ray_distance is None or d < ray_distance:
                    ray_distance = d

        # update the particle filter
        self.particle_filter.update(requested_motion, measurements=[ray_distance])

        if update_all or update_obstacles and update_particles and update_poses:
            self.canvas.delete("all")
            self.draw_grid()
            self.draw_particles(self.particle_filter.particles)
            self.draw_real_pose(self.real_pose)
            # self.draw_estimated_poses()
            self.draw_landmarks(self.particle_filter.landmarks)
            self.draw_boundaries(self.particle_filter.boundaries)
        elif update_particles or update_poses:
            self.canvas.delete("pose")
            self.canvas.delete("particle")
            self.draw_particles(self.particle_filter.particles)
            self.draw_real_pose(self.real_pose)
            # self.draw_estimated_poses()
        elif update_obstacles:
            self.canvas.delete("obstacle")
            self.canvas.delete("free")
            self.draw_grid()
        else:
            # print("No updates")
            pass

        # self.particle_filter.update(motion=(np.random.normal(0, 5), np.random.normal(0, 2 * np.pi)))
        self.root.after(1, self.update)

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
        :return: requested motion, (distance, angle) where distance is the distance to move and angle is the angle to turn
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
                    self.draw_box(i, j, color="white", tag="free")

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

    def draw_real_pose(self, pose):
        """
        Draw the poses on the canvas, each pose represent the estimated pose of a robot
        :param pose: List of poses to draw, each pose is a tuple (x, y, theta)
        :return:
        """
        x, y, theta, _ = pose
        self.canvas.create_oval((x * self.box_size) - (self.pose_size / 2),
                                (y * self.box_size) - (self.pose_size / 2),
                                (x * self.box_size) + (self.pose_size / 2),
                                (y * self.box_size) + (self.pose_size / 2),
                                fill="blue", outline="blue", tags="pose")
        # Draw a line to show the orientation of the pose
        self.canvas.create_line(x * self.box_size,
                                y * self.box_size,
                                (x * self.box_size) + (np.cos(theta) * self.pose_size * 2),
                                (y * self.box_size) + (np.sin(theta) * self.pose_size * 2),
                                fill="orange", tags="pose")

    def draw_estimated_poses(self, poses):
        """
        Draw the poses on the canvas, each pose represent the estimated pose of a robot
        :param poses: List of poses to draw, each pose is a tuple (x, y, theta)
        :return:
        """
        for pose in poses:
            x, y, theta = pose
            # Draw a blue dot for each pose
            self.canvas.create_oval((x * self.box_size) - (self.pose_size / 2),
                                    (y * self.box_size) - (self.pose_size / 2),
                                    (x * self.box_size) + (self.pose_size / 2),
                                    (y * self.box_size) + (self.pose_size / 2),
                                    fill="green", outline="green", tags="pose")
            # Draw a line to show the orientation of the pose
            self.canvas.create_line(x * self.box_size,
                                    y * self.box_size,
                                    (x * self.box_size) + (np.cos(theta) * self.pose_size * 2),
                                    (y * self.box_size) + (np.sin(theta) * self.pose_size * 2),
                                    fill="orange", tags="pose")

    def draw_particles(self, particles):
        """
        Draw the particles on the canvas
        :return: None
        """
        for particle in particles:
            x, y, theta, _ = particle
            # Draw a red dot for each particle
            # TODO: check the oval is being drawn correctly, it should be centered at (x, y)
            self.canvas.create_oval((x * self.box_size) - (self.particle_size / 2),
                                    (y * self.box_size) - (self.particle_size / 2),
                                    (x * self.box_size) + (self.particle_size / 2),
                                    (y * self.box_size) + (self.particle_size / 2),
                                    fill="red", outline="red", tags="particle")
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
                                    fill="orange", tags="boundary")

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
    lm = np.array([(10, 10, .25 * np.pi), (20, 20, .75 * np.pi), (30, 30, 1.25 * np.pi)])
    bounds = np.array([(0.5, 0.5, 0.5, 99.5),
                           (0.5, 0.5, 99.5, 0.5),
                           (99.5, 0.5, 99.5, 99.5),
                           (0.5, 99.5, 99.5, 99.5)])

    pf = ParticleFilter(m, 1000, landmarks=lm, boundaries=bounds)
    pf.start()

    gui = ParticleFilterGUI(pf)

    # while True:
    #     # pf.update()
    #     gui.update()

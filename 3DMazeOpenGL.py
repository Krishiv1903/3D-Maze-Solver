import numpy as np
import random
import math
from enum import Enum 
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import heapq
from collections import deque
import sys
import time

class Shape(Enum):
    CUBE = 1
    SPHERE = 3

class ShellMazeGenerator:
    @staticmethod
    def generate(size, wall_probability=0.4):
        """Generate a shell-only maze (outer layer only)"""
        diameter = 2 * size + 1
        maze = np.ones((diameter, diameter, diameter), dtype=np.int8)
        center = size

        # Mark all outer shell cells (0 = path, 1 = wall)
        for x in range(diameter):
            for y in range(diameter):
                for z in range(diameter):
                    dist = math.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
                    if size-0.5 <= dist <= size+0.5:  # Shell thickness
                        if random.random() > wall_probability:
                            maze[x, y, z] = 0  # Path
                    else:
                        maze[x, y, z] = -1  # Ignore (inner/outer void)

        # Set start (2) and end (3) on the shell
        open_positions = np.argwhere((maze == 0))
        start_idx, end_idx = random.sample(range(len(open_positions)), 2)
        start = tuple(open_positions[start_idx])
        end = tuple(open_positions[end_idx])
        maze[start] = 2
        maze[end] = 3

        return maze, start

class CubeMazeGenerator:
    @staticmethod
    def generate(size, wall_probability=0.4):
        """Generate a hollow cube maze (only outer layer)"""
        maze = np.ones((size, size, size), dtype=np.int8)  # Start with all walls
        
        # Mark outer layer (shell) cells
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    # Check if cell is on any face of the cube
                    if (x == 0 or x == size-1 or 
                        y == 0 or y == size-1 or 
                        z == 0 or z == size-1):
                        if random.random() > wall_probability:
                            maze[x, y, z] = 0  # Path
                    else:
                        maze[x, y, z] = -1  # Inner cells are invisible
        
        # Set start and end on outer layer
        open_positions = np.argwhere((maze == 0))
        start_idx, end_idx = random.sample(range(len(open_positions)), 2)
        start = tuple(open_positions[start_idx])
        end = tuple(open_positions[end_idx])
        maze[start] = 2  # Start
        maze[end] = 3    # End
        
        return maze, start

class MazeVisualizer:
    def __init__(self, maze, start_pos, shape=Shape.CUBE):
        self.start_time = time.time()
        self.game_complete = False
        self.maze = maze
        self.shape = shape
        self.size = maze.shape[0] // 2
        self.camera_pos = [start_pos[0]-self.size, start_pos[1]-self.size, start_pos[2]-self.size]
        self.camera_yaw = -90
        self.camera_pitch = 0
        self.camera_speed = 0.2
        self.mouse_sensitivity = 0.1
        self.first_mouse = True
        self.last_x, self.last_y = 0, 0
        self.wireframe = False
        self.path = []
        self.win_width, self.win_height = 800, 600
        self.fov = 60
        self.view_mode = "third_person"  # or "third_person"
        self.third_person_distance = 5
        self.third_person_angle_x = 30
        self.third_person_angle_y = 0
        self.third_camera_distance = self.size * 2
        self.player_pos = [start_pos[0]-self.size, start_pos[1]-self.size, start_pos[2]-self.size]

    def initialize(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.win_width, self.win_height)
        glutCreateWindow(b"3D Maze Visualizer")
        glDisable(GL_LIGHTING)  # Ensure lighting is off
        glDisable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutMotionFunc(self.mouse_look)
        glutPassiveMotionFunc(self.mouse_look)
        glutSpecialFunc(self.special_keys)
        glutReshapeFunc(self.reshape)
        glutMouseFunc(self.mouse_click)
        glutSetCursor(GLUT_CURSOR_NONE)

    def reshape(self, width, height):
        self.win_width = width
        self.win_height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, width/height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        if self.view_mode == "first_person":
            front = self.get_front_vector()
            center = [
                self.camera_pos[0] + front[0],
                self.camera_pos[1] + front[1],
                self.camera_pos[2] + front[2]
            ]
            gluLookAt(
                self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                center[0], center[1], center[2],
                0, 1, 0
            )
        else:  # third_person
            glTranslatef(0, 0, -self.third_camera_distance)
            glRotatef(self.third_person_angle_x, 1, 0, 0)
            glRotatef(self.third_person_angle_y, 0, 1, 0)
        
        self.draw_shell_maze()
        self.draw_player()
        glutSwapBuffers()

    def draw_player(self):
        if self.view_mode == "third_person":
            glPushMatrix()
            
            # Get grid position
            grid_x = round(self.player_pos[0] + self.size)
            grid_y = round(self.player_pos[1] + self.size)
            grid_z = round(self.player_pos[2] + self.size)
            
            # Calculate surface normal (direction to push player outward)
            normal = [0, 0, 0]
            if grid_x == 0: normal[0] = -1
            elif grid_x == self.maze.shape[0]-1: normal[0] = 1
            elif grid_y == 0: normal[1] = -1
            elif grid_y == self.maze.shape[1]-1: normal[1] = 1
            elif grid_z == 0: normal[2] = -1
            else: normal[2] = 1
            
            # Apply outward offset (0.45 units along normal)
            adjusted_pos = [
                self.player_pos[0] + normal[0] * 0.45,
                self.player_pos[1] + normal[1] * 0.45,
                self.player_pos[2] + normal[2] * 0.45
            ]
            
            glTranslatef(*adjusted_pos)
            
            # Rest of the drawing code...
            lighting_was_enabled = glIsEnabled(GL_LIGHTING)
            if lighting_was_enabled:
                glDisable(GL_LIGHTING)
            
            glColor4f(0.0, 0.0, 1.0, 1.0)
            glutSolidCube(0.9)
            
            if lighting_was_enabled:
                glEnable(GL_LIGHTING)
            glPopMatrix()
    
    def draw_shell_maze(self):
        center = self.size
        diameter = self.maze.shape[0]

        for x in range(diameter):
            for y in range(diameter):
                for z in range(diameter):
                    if self.maze[x, y, z] >= 0:  # Only draw shell (ignore -1)
                        cell_type = self.maze[x, y, z]
                        gl_x, gl_y, gl_z = x - center, y - center, z - center

                        # Set colors
                        if cell_type == 2:
                            color = (0.0, 1.0, 0.0, 1.0)  # Green (start)
                        elif cell_type == 3:
                            color = (1.0, 0.0, 0.0, 1.0)  # Red (end)
                        elif (x, y, z) in self.path:
                            color = (0.5, 0.5, 1.0, 1.0)  # Light blue (path)
                        elif cell_type == 1:
                            color = (1.0, 1.0, 1.0, 1.0)  # White (wall)
                        else:
                            color = (0.7, 0.7, 0.7, 0.5)  # Gray (path)

                        # Draw cell
                        glPushMatrix()
                        glTranslatef(gl_x, gl_y, gl_z)
                        # if cell_type == 1:
                        #     glScalef(1.0, 1.0, 1.2)  # Walls have depth
                        glColor4fv(color)
                        glutSolidCube(0.9)
                        glPopMatrix()

    def mouse_click(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            print("\nCalculating shortest path...")
            self.find_path()

    def mouse_look(self, x, y):
        if self.first_mouse:
            self.last_x = x
            self.last_y = y
            self.first_mouse = False
            return
        # Calculate offsets
        x_offset = x - self.win_width // 2
        y_offset = self.win_height // 2 - y  # Inverted Y-axis

        # Apply sensitivity
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        if self.view_mode == "first_person":
            self.camera_yaw += x_offset
            self.camera_pitch += y_offset
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))

        # Warp cursor to center (without recursion)
        if (x, y) != (self.win_width // 2, self.win_height // 2):
            glutWarpPointer(self.win_width // 2, self.win_height // 2)
            self.last_x = self.win_width // 2
            self.last_y = self.win_height // 2

        glutPostRedisplay()
    
    def keyboard(self, key, x, y):
        key = key.decode('utf-8')
        if key == 'w':
            self.move_forward()
        elif key == 's':
            self.move_backward()
        elif key == 'd':
            self.move_right()
        elif key == 'a':
            self.move_left()
        elif key == 'r':
            self.move_downward()
        elif key == 'e':
            self.move_upward()
        elif key == 'f':
            self.toggle_view_mode()
        elif key == 'q':
            glutTimerFunc(500, self.close_window, 0)
        glutPostRedisplay()

    def special_keys(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.third_person_angle_y -= 5
        elif key == GLUT_KEY_RIGHT:
            self.third_person_angle_y += 5
        elif key == GLUT_KEY_UP:
            self.third_person_angle_x -= 5
        elif key == GLUT_KEY_DOWN:
            self.third_person_angle_x += 5
        elif key == GLUT_KEY_PAGE_UP:
            self.third_camera_distance = max(5, self.third_camera_distance - 1)
        elif key == GLUT_KEY_PAGE_DOWN:
            self.third_camera_distance += 1
        glutPostRedisplay()

    def toggle_view_mode(self):
        self.view_mode = "third_person" if self.view_mode == "first_person" else "first_person"
        glutPostRedisplay()

    def snap_to_face(self, pos):
            """Ensure player stays properly aligned to cube faces"""
            grid_x = round(pos[0] + self.size)
            grid_y = round(pos[1] + self.size)
            grid_z = round(pos[2] + self.size)
            
            # Check which face we're moving toward
            if grid_x <= 0:
                pos[0] = -self.size
            elif grid_x >= self.maze.shape[0]-1:
                pos[0] = self.size
            elif grid_y <= 0:
                pos[1] = -self.size
            elif grid_y >= self.maze.shape[1]-1:
                pos[1] = self.size
            elif grid_z <= 0:
                pos[2] = -self.size
            else:
                pos[2] = self.size
                
            return pos

    def move_forward(self):
        if self.game_complete:
            return
    
        front = self.get_front_vector()
        move_vec = [x * self.camera_speed for x in front]
        
        new_player_pos = [
            self.player_pos[0] + move_vec[0],
            self.player_pos[1] + move_vec[1],
            self.player_pos[2] + move_vec[2]
        ]
        
        # Snap to new face if transitioning
        new_player_pos = self.snap_to_face(new_player_pos)
        
        if self.is_valid_position(new_player_pos):
            self.player_pos = new_player_pos
            if self.view_mode == "first_person":
                self.camera_pos = new_player_pos
            if self.check_win_condition(new_player_pos):
                self.handle_win()

    def move_backward(self):
        if self.game_complete:
            return
        
        front = self.get_front_vector()
        move_vec = [x * -self.camera_speed for x in front]
        
        new_player_pos = [
            self.player_pos[0] + move_vec[0],
            self.player_pos[1] + move_vec[1],
            self.player_pos[2] + move_vec[2]
        ]
        
        # Snap to new face if transitioning
        new_player_pos = self.snap_to_face(new_player_pos)
        
        if self.is_valid_position(new_player_pos):
            self.player_pos = new_player_pos
            if self.view_mode == "first_person":
                self.camera_pos = new_player_pos
            if self.check_win_condition(new_player_pos):
                self.handle_win()

    def move_left(self):
        if self.game_complete:
            return
        
        front = self.get_front_vector()
        right = [front[2], 0, -front[0]]
        norm = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
        if norm > 0:
            right = [x/norm * -self.camera_speed for x in right]
            
            new_player_pos = [
                self.player_pos[0] + right[0],
                self.player_pos[1] + right[1],
                self.player_pos[2] + right[2]
            ]
            
            # Snap to new face if transitioning
            new_player_pos = self.snap_to_face(new_player_pos)
            
            if self.is_valid_position(new_player_pos):
                self.player_pos = new_player_pos
                if self.view_mode == "first_person":
                    self.camera_pos = new_player_pos
                if self.check_win_condition(new_player_pos):
                    self.handle_win()

    def move_right(self):
        if self.game_complete:
            return
        
        front = self.get_front_vector()
        right = [front[2], 0, -front[0]]
        norm = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
        if norm > 0:
            right = [x/norm * self.camera_speed for x in right]
            
            new_player_pos = [
                self.player_pos[0] + right[0],
                self.player_pos[1] + right[1],
                self.player_pos[2] + right[2]
            ]
            
            # Snap to new face if transitioning
            new_player_pos = self.snap_to_face(new_player_pos)
            
            if self.is_valid_position(new_player_pos):
                self.player_pos = new_player_pos
                if self.view_mode == "first_person":
                    self.camera_pos = new_player_pos
                if self.check_win_condition(new_player_pos):
                    self.handle_win()

    def move_upward(self):
        if self.game_complete:
            return
        
        front = self.get_front_vector()
        # Calculate upward vector (perpendicular to front and right vectors)
        right = [front[2], 0, -front[0]]  # Same right vector calculation as in move_left/right
        up = [
            right[1] * front[2] - right[2] * front[1],
            right[2] * front[0] - right[0] * front[2],
            right[0] * front[1] - right[1] * front[0]
        ]
        
        # Normalize the up vector
        norm = math.sqrt(up[0]**2 + up[1]**2 + up[2]**2)
        if norm > 0:
            up = [x/norm * self.camera_speed for x in up]
            
            new_player_pos = [
                self.player_pos[0] + up[0],
                self.player_pos[1] + up[1],
                self.player_pos[2] + up[2]
            ]
               
            # Snap to new face if transitioning
            new_player_pos = self.snap_to_face(new_player_pos)   
                
            if self.is_valid_position(new_player_pos):
                self.player_pos = new_player_pos
                if self.view_mode == "first_person":
                    self.camera_pos = new_player_pos
                if self.check_win_condition(new_player_pos):
                    self.handle_win()

    def move_downward(self):
        if self.game_complete:
            return
        
        front = self.get_front_vector()
        # Calculate upward vector first (same as in move_upward)
        right = [front[2], 0, -front[0]]
        up = [
            right[1] * front[2] - right[2] * front[1],
            right[2] * front[0] - right[0] * front[2],
            right[0] * front[1] - right[1] * front[0]
        ]
        
        # Normalize and invert for downward movement
        norm = math.sqrt(up[0]**2 + up[1]**2 + up[2]**2)
        if norm > 0:
            down = [x/norm * -self.camera_speed for x in up]
            
            new_player_pos = [
                self.player_pos[0] + down[0],
                self.player_pos[1] + down[1],
                self.player_pos[2] + down[2]
            ]
            
            # Snap to new face if transitioning
            new_player_pos = self.snap_to_face(new_player_pos)
            
            if self.is_valid_position(new_player_pos):
                self.player_pos = new_player_pos
                if self.view_mode == "first_person":
                    self.camera_pos = new_player_pos
                if self.check_win_condition(new_player_pos):
                    self.handle_win()

    def get_front_vector(self):
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)
        front_x = math.cos(yaw_rad) * math.cos(pitch_rad)
        front_y = math.sin(pitch_rad)
        front_z = math.sin(yaw_rad) * math.cos(pitch_rad)
        return [front_x, front_y, front_z]

    def is_valid_position(self, pos):
        grid_x = round(pos[0] + self.size)
        grid_y = round(pos[1] + self.size)
        grid_z = round(pos[2] + self.size)
        
        if (0 <= grid_x < self.maze.shape[0] and 
            0 <= grid_y < self.maze.shape[1] and 
            0 <= grid_z < self.maze.shape[2]):
            return self.maze[grid_x, grid_y, grid_z] != 1
        return False

    # A star
    def find_path(self):
        start_time = time.time()    # Start timing
        
        start_pos = (round(self.player_pos[0] + self.size), 
                    round(self.player_pos[1] + self.size), 
                    round(self.player_pos[2] + self.size))
        end_pos = tuple(np.argwhere(self.maze == 3)[0])
        open_set = [(0 + self._heuristic(start_pos, end_pos), 0, start_pos, [start_pos])]
        came_from = {}
        g_scores = {start_pos: 0}

        while open_set:
            _, g_score, current, path = heapq.heappop(open_set)
            if current == end_pos:
                end_time = time.time()  # Stop timing
                self.path = path
                
                # Print results to terminal
                print("\nPATH FOUND SUCCESSFULLY!")
                print(f"Pathfinding time: {(end_time - start_time)*1000:.2f} ms")
                print(f"Path length: {len(path)} steps")
                
                # Display path
                glutPostRedisplay()  # Ensure path is visible
                return path

            for neighbor in self._get_shell_neighbors(current):
                tentative_g = g_score + 1
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end_pos)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
        
        print("\nPATHFINDING FAILED: No path exists")
        glutTimerFunc(100, self.close_window, 0)
        return []
    
    # A star bidirectional
    # def find_path(self):
    #     start_time = time.time()
        
    #     start_pos = (round(self.player_pos[0] + self.size), 
    #                 round(self.player_pos[1] + self.size), 
    #                 round(self.player_pos[2] + self.size))
    #     end_pos = tuple(np.argwhere(self.maze == 3)[0])
        
    #     # Initialize forward and backward searches
    #     forward_open = [(self._heuristic(start_pos, end_pos), 0, start_pos)]
    #     backward_open = [(self._heuristic(end_pos, start_pos), 0, end_pos)]
        
    #     forward_g = {start_pos: 0}
    #     backward_g = {end_pos: 0}
        
    #     forward_parent = {start_pos: None}
    #     backward_parent = {end_pos: None}
        
    #     intersect_node = None
        
    #     while forward_open and backward_open:
    #         # Forward search
    #         _, g_f, current_f = heapq.heappop(forward_open)
    #         for neighbor in self._get_shell_neighbors(current_f):
    #             new_g = g_f + 1
    #             if neighbor not in forward_g or new_g < forward_g[neighbor]:
    #                 forward_g[neighbor] = new_g
    #                 forward_parent[neighbor] = current_f
    #                 heapq.heappush(forward_open, (new_g + self._heuristic(neighbor, end_pos), new_g, neighbor))
    #                 if neighbor in backward_g:
    #                     intersect_node = neighbor
    #                     break
            
    #         # Backward search
    #         _, g_b, current_b = heapq.heappop(backward_open)
    #         for neighbor in self._get_shell_neighbors(current_b):
    #             new_g = g_b + 1
    #             if neighbor not in backward_g or new_g < backward_g[neighbor]:
    #                 backward_g[neighbor] = new_g
    #                 backward_parent[neighbor] = current_b
    #                 heapq.heappush(backward_open, (new_g + self._heuristic(neighbor, start_pos), new_g, neighbor))
    #                 if neighbor in forward_g:
    #                     intersect_node = neighbor
    #                     break
            
    #         if intersect_node:
    #             break
        
    #     end_time = time.time()
    #     if not intersect_node:
    #         print("\nPATHFINDING FAILED: No path exists")
    #         glutTimerFunc(100, self.close_window, 0)
    #         return []
        
    #     # Reconstruct path
    #     path = []
    #     node = intersect_node
    #     while node:
    #         path.insert(0, node)
    #         node = forward_parent[node]
        
    #     node = backward_parent[intersect_node]
    #     while node:
    #         path.append(node)
    #         node = backward_parent[node]
        
    #     self.path = path
    #     # Print results to terminal
    #     print("\nPATH FOUND SUCCESSFULLY!")
    #     print(f"Pathfinding time: {(end_time - start_time)*1000:.2f} ms")
    #     print(f"Path length: {len(path)} steps")
    #     glutPostRedisplay()  # Ensure path is visible
    #     return path

    # Dijkstra
    # def find_path(self):
    #     start_time = time.time()
    #     start_pos = (round(self.player_pos[0] + self.size), 
    #                 round(self.player_pos[1] + self.size), 
    #                 round(self.player_pos[2] + self.size))
    #     end_pos = tuple(np.argwhere(self.maze == 3)[0])
        
    #     open_set = [(0, start_pos, [start_pos])]
    #     visited = set()
        
    #     while open_set:
    #         cost, current, path = heapq.heappop(open_set)
    #         if current == end_pos:
    #             end_time = time.time()
    #             self.path = path
    #             print("\nDijkstra's Algorithm")
    #             print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    #             print(f"Path length: {len(path)} steps")
    #             glutPostRedisplay()
    #             return path
                
    #         if current in visited:
    #             continue
    #         visited.add(current)
            
    #         for neighbor in self._get_shell_neighbors(current):
    #             if neighbor not in visited:
    #                 heapq.heappush(open_set, (cost + 1, neighbor, path + [neighbor]))
        
    #     end_time = time.time()
    #     print("\nPATHFINDING FAILED: No path exists")
    #     glutTimerFunc(100, self.close_window, 0)
    #     return []

    # BFS
    # def find_path(self):
    #     start_time = time.time()
    #     start_pos = (round(self.player_pos[0] + self.size), 
    #                 round(self.player_pos[1] + self.size), 
    #                 round(self.player_pos[2] + self.size))
    #     end_pos = tuple(np.argwhere(self.maze == 3)[0])
        
    #     queue = deque([(start_pos, [start_pos])])
    #     visited = set()
        
    #     while queue:
    #         current, path = queue.popleft()
    #         if current == end_pos:
    #             end_time = time.time()
    #             self.path = path
    #             print("\nBreadth-First Search (BFS)")
    #             print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    #             print(f"Path length: {len(path)} steps")
    #             glutPostRedisplay()
    #             return path
                
    #         if current in visited:
    #             continue
    #         visited.add(current)
            
    #         for neighbor in self._get_shell_neighbors(current):
    #             if neighbor not in visited:
    #                 queue.append((neighbor, path + [neighbor]))
        
    #     end_time = time.time()
    #     print("\nPATHFINDING FAILED: No path exists")
    #     glutTimerFunc(100, self.close_window, 0)
    #     return []

    # Greedy BFS
    # def find_path(self):
    #     start_time = time.time()
    #     start_pos = (round(self.player_pos[0] + self.size), 
    #                 round(self.player_pos[1] + self.size), 
    #                 round(self.player_pos[2] + self.size))
    #     end_pos = tuple(np.argwhere(self.maze == 3)[0])
        
    #     open_set = [(self._heuristic(start_pos, end_pos), start_pos, [start_pos])]
    #     visited = set()
        
    #     while open_set:
    #         _, current, path = heapq.heappop(open_set)
    #         if current == end_pos:
    #             end_time = time.time()
    #             self.path = path
    #             print(f"\nPATH FOUND SUCCESSFULLY!")
    #             print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    #             print(f"Path length: {len(path)} steps")
    #             glutPostRedisplay()
    #             return path
                
    #         if current in visited:
    #             continue
    #         visited.add(current)
            
    #         for neighbor in self._get_shell_neighbors(current):
    #             if neighbor not in visited:
    #                 heapq.heappush(open_set, (self._heuristic(neighbor, end_pos), neighbor, path + [neighbor]))
        
    #     end_time = time.time()
    #     print("\nPATHFINDING FAILED: No path exists")
    #     glutTimerFunc(100, self.close_window, 0)
    #     return []

    # DFS   
    # def find_path(self):
    #     start_time = time.time()
    #     start_pos = (round(self.player_pos[0] + self.size), 
    #                 round(self.player_pos[1] + self.size), 
    #                 round(self.player_pos[2] + self.size))
    #     end_pos = tuple(np.argwhere(self.maze == 3)[0])
        
    #     stack = [(start_pos, [start_pos])]
    #     visited = set()
        
    #     while stack:
    #         current, path = stack.pop()
    #         if current == end_pos:
    #             end_time = time.time()
    #             self.path = path
    #             print("\nDepth-First Search (DFS)")
    #             print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    #             print(f"Path length: {len(path)} steps")
    #             glutPostRedisplay()
    #             return path
                
    #         if current in visited:
    #             continue
    #         visited.add(current)
            
    #         for neighbor in self._get_shell_neighbors(current):
    #             if neighbor not in visited:
    #                 stack.append((neighbor, path + [neighbor]))
        
    #     end_time = time.time()
    #     print("\nPATHFINDING FAILED: No path exists")
    #     glutTimerFunc(100, self.close_window, 0)
    #     return []

    def _get_shell_neighbors(self, pos):
        x, y, z = pos
        size = self.maze.shape[0]
        neighbors = []
        
        # Check if current position is on outer layer
        if not (x == 0 or x == size-1 or y == 0 or y == size-1 or z == 0 or z == size-1):
            return neighbors
        
        # Check all 6 possible directions (including face transitions)
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Boundary check - if we go out of bounds, we've reached an edge
            if not (0 <= nx < size and 0 <= ny < size and 0 <= nz < size):
                continue
                
            # The neighbor must be on the outer layer and walkable
            if (nx == 0 or nx == size-1 or ny == 0 or ny == size-1 or nz == 0 or nz == size-1) \
            and self.maze[nx, ny, nz] != 1:
                neighbors.append((nx, ny, nz))
        
        # Special case: handle corner transitions (where three faces meet)
        # This allows movement around corners while staying on the surface
        if (x == 0 or x == size-1) and (y == 0 or y == size-1) and (z == 0 or z == size-1):
            # We're at a corner - add diagonal moves that stay on the surface
            for dx, dy, dz in [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                            (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                            (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                
                if (0 <= nx < size and 0 <= ny < size and 0 <= nz < size):
                    if (nx == 0 or nx == size-1 or ny == 0 or ny == size-1 or nz == 0 or nz == size-1) \
                    and self.maze[nx, ny, nz] != 1:
                        neighbors.append((nx, ny, nz))
        
        return neighbors
    
    #Euclidean 
    # def _heuristic(self, a, b):
    #     return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    
    # Manhattan
    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    # Chebyshev
    # def _heuristic(self, a, b):
    #     return max(abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2]))

    def check_win_condition(self, new_pos):
        grid_x = round(new_pos[0] + self.size)
        grid_y = round(new_pos[1] + self.size)
        grid_z = round(new_pos[2] + self.size)
        
        if (0 <= grid_x < self.maze.shape[0] and 
            0 <= grid_y < self.maze.shape[1] and 
            0 <= grid_z < self.maze.shape[2]):
            if self.maze[grid_x, grid_y, grid_z] == 3:  # Red end block
                return True
        return False

    def close_window(self, value):
        glutLeaveMainLoop()
        # Alternative if glutLeaveMainLoop doesn't work:
        # glutDestroyWindow(glutGetWindow())

    def handle_win(self):
        self.game_complete = True
        end_time = time.time()
        time_taken = end_time - self.start_time
        
        # Print to terminal
        print("\n\nMAZE COMPLETED SUCCESSFULLY!")
        print(f"Time taken: {time_taken:.2f} seconds")
        print("Congratulations!\n")
        
        # Close the window after a short delay
        glutLeaveMainLoop()
        # If the above doesn't work, try:
        # glutDestroyWindow(glutGetWindow())

    def run(self):
        try:
            glutMainLoop()
        except:
            pass  # Gracefully handle window closure

if __name__ == "__main__":
    maze, start_pos = CubeMazeGenerator.generate(size=40)
    visualizer = MazeVisualizer(maze, start_pos)
    visualizer.initialize()
    visualizer.run()
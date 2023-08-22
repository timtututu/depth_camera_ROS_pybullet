import pygame
import math
from queue import PriorityQueue
import pybullet as p
import pybullet_data
import math
import numpy as np
import time
import random
import json
import matplotlib.pyplot as plt


# PYBULLET

x = 0
y = 0
z = 0
heading = 0


def updatePosition():
    global x, y, z, heading
    pos, hquat = p.getBasePositionAndOrientation(car)
    heading = p.getEulerFromQuaternion(hquat)[2]
    x = pos[0]
    y = pos[1]
    z = pos[2]

# Tuning Parameters
translation = [-3, -5]
stretching = [0.8, 1]
gap_threshold = 40  # min gap a car fits through
car_buffer = 20  # close obstacles
steering_constant = 0.005  # nerf correction
frames = 2  # takes snapshots every {frames} seconds
capture = True  # boolean to save images

zone_width = 30  # obstacle meters
zone_depth = 30  # obstacle meters
num_cubes = 30  # Number of cubes to load

dimensions = 50  # resolution
scale = 0.16  # pygame to world coordinates

referencePoint = [0, 0]

# Physics
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.resetSimulation()
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)  # either this

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=0.1,
    cameraYaw=- 90,
    cameraPitch=10,
    cameraTargetPosition=[
        0, 0, 0.2],
    physicsClientId=0
)
# Path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Assets
p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf")

# Define the race position
race_position = [0, 0, 0]  # [x, y, z]

# Init
start_time = time.time()
elapsed_time = 0

BARRIER = [[2, 0]]

cylinder_shape = p.createCollisionShape(
    p.GEOM_CYLINDER, radius=0.15, height=0.5)

for coordinate in BARRIER:
    p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=cylinder_shape,
        basePosition=[coordinate[0], coordinate[1], 0.3],
    )

# Wheels
inactive_wheels = [5, 7]
wheels = [2, 3]

for wheel in inactive_wheels:
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
steering = [4, 6]


BARRIER = []  # List to hold generated barriers
BARRIER_IDS = {}  # Dictionary to hold generated barriers along with their IDs
MAX_OBSTACLES = 40

# Cylinder shape for the barriers
cylinder_shape = p.createCollisionShape(
    p.GEOM_CYLINDER, radius=0.15, height=0.5)

# Function to check if a new barrier is too close to existing ones
def too_close(new_barrier, existing_barriers, min_distance=1):
    for barrier in existing_barriers:
        dist = math.sqrt((new_barrier[0]-barrier[0])
                         ** 2 + (new_barrier[1]-barrier[1])**2)
        if dist < min_distance:
            return True
    return False

def spawnObstaclesAroundCar():
    new_barrier_polar = [random.uniform(4, 9), random.uniform(-0.4, 0.4)]
    new_barrier_cartesian = [new_barrier_polar[0] * math.cos(new_barrier_polar[1]) + x,
                             new_barrier_polar[0] * math.sin(new_barrier_polar[1]) + y]
    # global BARRIER
    if not too_close(new_barrier_cartesian, BARRIER):
        BARRIER.append(new_barrier_cartesian)

        # Generate the barriers in the simulation
        obstacle_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=cylinder_shape,
            basePosition=[new_barrier_cartesian[0],
                          new_barrier_cartesian[1], 0.3],
        )

        # Add the obstacle id to the dictionary
        BARRIER_IDS[tuple(new_barrier_cartesian)] = obstacle_id

        if len(BARRIER) > MAX_OBSTACLES:
            # Remove the oldest obstacle
            oldest_obstacle = BARRIER.pop(0)
            p.removeBody(BARRIER_IDS[tuple(oldest_obstacle)])
            del BARRIER_IDS[tuple(oldest_obstacle)]

# DEPTH CAMERA
def depth_scan(i):
    updatePosition()
    global x, y, z, heading
    p.resetDebugVisualizerCamera(
        cameraDistance=0.1,
        cameraYaw=math.degrees(heading) - 90,
        cameraPitch=10,
        cameraTargetPosition=[x + math.cos(heading), y + math.sin(heading), z + 0.2],
        physicsClientId=0
    )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    cam_info = p.getDebugVisualizerCamera()
    view_matrix = cam_info[2]
    proj_matrix = cam_info[3]
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        256, 256, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )
    depth = np.array(depth_img).reshape((256, 256))

    # POINT CLOUD
    projection = np.array(proj_matrix).reshape(4, 4)
    fx = projection[0][0]
    fy = -projection[1][1]
    cx = projection[0][2]
    cy = projection[1][2]

    # Generate point cloud
    v, u = np.indices(depth.shape)
    z = depth.copy()
    valid_mask = (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy + 150
    point_cloud = np.column_stack(
        (x[valid_mask], y[valid_mask], 10000 - z[valid_mask] * 10000))

    # Convert the angle to radians
    theta = math.radians(12)

    # Define the rotation matrix around the x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    rotated_points = np.dot(point_cloud, R_x)

    # Filter background and ground
    obstacles = rotated_points[(rotated_points[:, 2] > 0)
                               & (rotated_points[:, 1] > 0.5)]

    # 2D map
    map = obstacles.copy()
    map[:, 1], map[:, 2] = obstacles[:, 2], obstacles[:, 1]
    map[:, 2] = 0
    map = map[:, :2]

    # Transform map
    scaled_map = map * ((dimensions - 1) / 238)  # scale it down
    local_map = np.array([[((dimensions - 1) / 2) -
                           (x - ((dimensions - 1) / 2)), y] for x, y in scaled_map]).astype(np.int64)  # flip the y axis along its middle

    local_map[:, 0] = (local_map[:, 0] - ((dimensions - 1) / 2)
                       ) * stretching[0] + ((dimensions - 1) / 2)
    local_map[:, 1] = (local_map[:, 1] - ((dimensions - 1) / 2)
                       ) * stretching[1] + ((dimensions - 1) / 2)

    local_map[:, 0] += translation[0]
    local_map[:, 1] += translation[1]

    # Check bounds
    x_condition = (local_map[:, 0] >= 0) & (local_map[:, 0] <= 24)
    y_condition = (local_map[:, 1] >= 0) & (local_map[:, 1] <= 24)

    # Combine the conditions for x and y using logical AND (&) to get the final condition
    final_condition = x_condition & y_condition

    # Apply boolean indexing to get the filtered array
    local_map = local_map[final_condition]

    # Save image
    plt.imsave(f'autopilot_img/depth_{i}.png', depth)
    save_map(local_map, f'autopilot_img/local_map_{i}.png')
    return local_map

# Save local_map
def save_map(array, filename):
    plt.clf()
    plt.scatter(array[:, 0], array[:, 1], s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, dimensions - 1)
    plt.ylim(0, dimensions - 1)
    plt.title('Obstacles')
    plt.savefig(filename)
    plt.close()

# Find optimal target
def find_target(map_points, threshold):
    map_points = np.array(map_points)
    map_points = np.vstack([map_points, [0, 0], [dimensions - 1, 0]])
    sorted_points = map_points[np.argsort(map_points[:, 0])]

    gaps = []
    for j in range(len(sorted_points) - 1):
        curr_point = sorted_points[j]
        next_point = sorted_points[j + 1]
        gap_size = next_point[0] - curr_point[0]

        if gap_size > threshold:
            gap_center = (curr_point[0] + next_point[0]) / 2
            gaps.append(gap_center)

    gaps = np.array(gaps)
    if(len(gaps)) > 0:
        optimal_gap = (int) (gaps[np.argmin(np.abs(gaps - ((dimensions - 1) // 2)))])
    else:
        optimal_gap = (int) ((dimensions - 1) // 2) # default center
    print(optimal_gap)
    return optimal_gap


# ASTAR
WIDTH = 500
ROWS = 50
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_inflation(self):
        return self.color == GREY

    def is_barrierinflation(self):
        return self.color == BLACK or self.color == GREY

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_inflation(self):
        self.color = GREY

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrierinflation():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrierinflation():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

        # DOWN LEFT
        if (self.row < self.total_rows - 1) and (not grid[self.row + 1][self.col].is_barrierinflation()) and (self.col > 0) and (not grid[self.row][self.col - 1].is_barrierinflation()):
            self.neighbors.append(grid[self.row + 1][self.col - 1])

        # DOWN RIGHT
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrierinflation() and self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():
            self.neighbors.append(grid[self.row + 1][self.col + 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation() and self.col > 0 and not grid[self.row][self.col - 1].is_barrierinflation():  # UP LEFT
            self.neighbors.append(grid[self.row - 1][self.col - 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation() and self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():  # UP RIGHT
            self.neighbors.append(grid[self.row - 1][self.col + 1])

    def __lt__(self, other):
        return False


def inflate_obstacles(grid, inflate_range=1):
    inflation = []
    for row in grid:
        for spot in row:
            if spot.is_barrier():
                inflation.append(spot)

    for spot in inflation:
        obstacle_row, obstacle_col = spot.get_pos()
        for row in range(-inflate_range, inflate_range + 1):
            for col in range(-inflate_range, inflate_range + 1):
                if row == 0 and col == 0:  # Skip the obstacle itself
                    continue
                inflated_row, inflated_col = obstacle_row + row, obstacle_col + col
                if (0 <= inflated_row < len(grid) and
                    0 <= inflated_col < len(grid[0]) and
                        not grid[inflated_row][inflated_col].is_barrier()):  # Make sure spot is in the grid and is not already an obstacle.
                    # Mark the spot as an obstacle
                    grid[inflated_row][inflated_col].make_inflation()


def h(p1, p2):  # manhattan vs eudclidean distance: heuristic function
    x1, y1 = p1
    x2, y2 = p2
    # return abs(x1 - x2) + abs(y1 - y2)
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


# draw path IMPORTANT BECAUSE ROBOT NEED TO FOLLOW THIS
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        PATH.append([scale * (current.y / (WIDTH/ROWS)),
                    scale * ((current.x / (WIDTH/ROWS)) - (ROWS//2))])
        draw()

def reset_path(grid, start, end):
    for row in grid:
        for spot in row:
            if spot != start and spot != end and not spot.is_barrier() and not spot.is_inflation():
                spot.reset()

def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))  # add to priority queue
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0  # distance from start
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())  # distance to end

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # way to exit the loop

        current = open_set.get()[2]  # starting at the start node
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if abs(neighbor.row - current.row) == 1 and abs(neighbor.col - current.col) == 1:
                temp_g_score = g_score[current] + math.sqrt(2)
            else:
                temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False

def make_grid(rows, width):  # makes grid
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid  # 2 dimensional list

def draw_grid(win, rows, width):  # draws gridlines
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)  # draws each box color
    draw_grid(win, rows, width)  # draws gridlines
    pygame.display.update()  # update display

def main(win, width):
    grid = make_grid(ROWS, width)
    start = None
    end = None

    draw(win, grid, ROWS, width)

    for coordinate in local_map:
        grid[coordinate[0]][coordinate[1]].make_barrier()

    reset_path(grid, start, end)
    inflate_obstacles(grid)  # Inflate obstacles before running the algorithm
    
    array_2d = []
    for row in grid:
        for spot in row:
            if spot.is_barrierinflation():
                obstacle_row, obstacle_col = spot.get_pos()
                array_2d.append([(obstacle_row - 1), 0])
                # array_2d[obstacle_row - 1, obstacle_col - 1] = 1
    np.set_printoptions(threshold=np.inf)
    # print(array_2d)

    
    start = grid[ROWS//2][0]
    start.make_start()
    end = grid[find_target(array_2d, 4)][ROWS//2]
    # end = grid[ROWS//2][ROWS//2]
    end.make_end()
    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)
    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)


def setReferencePointAsCurrentPosition():
    updatePosition()
    global referencePoint
    referencePoint = [x, y]

def moveTo(targetX, targetY):
    updatePosition()
    globalTargetX = targetX + referencePoint[0]
    globalTargetY = targetY + referencePoint[1]
    distance = math.sqrt((globalTargetX - x)**2 + (globalTargetY - y)**2)
    theta = math.atan2((globalTargetY - y), (globalTargetX - x))

    while distance > 1:
        updatePosition()
        spawnObstaclesAroundCar()
        # spawnObstaclesAroundCar()
        # spawnObstaclesAroundCar()
        distance = math.sqrt((globalTargetX - x)**2 + (globalTargetY - y)**2)
        theta = math.atan2((globalTargetY - y), (globalTargetX - x))

        maxForce = 20
        targetVelocity = 10*distance

        # velocity cap
        if targetVelocity > 15:
            targetVelocity = 15

        steeringAngle = theta - heading
        if abs(steeringAngle) > (math.pi):
            steeringAngle = heading - theta
        else:
            steeringAngle = theta - heading
        view_matrix_car = p.computeViewMatrix(
            cameraEyePosition=[
                x + 0.5*math.cos(heading), y + 0.5*math.sin(heading), z + 0.1],
            cameraTargetPosition=[
                x + 2*math.cos(heading), y + 2*math.sin(heading), z + 0.05],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix_car = p.computeProjectionMatrixFOV(
            fov=80,  # field of view
            aspect=1.0,  # aspect ratio
            nearVal=0.1,  # near clipping plane
            farVal=100.0  # far clipping plane
        )

        img_arr = p.getCameraImage(256, 256, viewMatrix=view_matrix_car,
                                   projectionMatrix=projection_matrix_car, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        depth_buf = np.reshape(img_arr[3], (256, 256))
        depth_img = 1 - np.array(depth_buf)
        firsthalf = depth_img[0][:128]
        totalfirsthalf = sum(firsthalf)
        secondhalf = depth_img[0][128:]
        totalsecondhalf = sum(secondhalf)
        middle = depth_img[0][54:202]
        totalmiddle = sum(middle)

        steer_reset_speed = 0.0001

        if steeringAngle != 0:
            if steeringAngle > 0:
                steeringAngle = max(0, steeringAngle - steer_reset_speed)
            elif steeringAngle < 0:
                steeringAngle = min(0, steeringAngle + steer_reset_speed)

        if totalmiddle > 0:
            if totalfirsthalf > totalsecondhalf:
                steeringAngle = steeringAngle - 1
                targetVelocity = targetVelocity + 5
            else:
                steeringAngle = steeringAngle + 1
                targetVelocity = targetVelocity + 5


        for wheel in wheels:
            p.setJointMotorControl2(car,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)

        for steer in steering:
            p.setJointMotorControl2(
                car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=-90,
            cameraPitch=-45,
            cameraTargetPosition=[x + 5, y, z],
            physicsClientId=0
        )

        p.stepSimulation()
        time.sleep(0.001)

def straightenOut():
    while abs(heading) > 0.1:
        updatePosition()
        for steer in steering:
            p.setJointMotorControl2(
                car, steer, p.POSITION_CONTROL, targetPosition=(0 - heading))
            
def stop():
    for wheel in wheels:
            p.setJointMotorControl2(car,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=0,
                                    force=20)


# AUTOPILOT
i = 0
while (True):
    local_map = depth_scan(i)
    i += 1
    PATH = []
    local_map[:, 1] *= -1
    local_map[:, 1] += 24
    local_map[:, 0] *= 2
    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption("A* Path Finding Algorithm")
    grid = main(WIN, WIDTH)
    print(grid)
    PATH.reverse()
    # p.createMultiBody(
    #     baseMass=1.0,
    #     baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=6),
    #     basePosition=[PATH[-1][0] + referencePoint[0], PATH[-1][1] + referencePoint[1], 3],
    # )
    setReferencePointAsCurrentPosition()
    for index, coordinate in enumerate(PATH):
        moveTo(coordinate[0], coordinate[1])
    straightenOut()
    stop()
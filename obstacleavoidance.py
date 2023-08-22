import os, inspect
import pybullet as p
import pybullet_data
import math
import time
import numpy as np
import random

# Car coordinates
x = 0
y = 0
z = 0
heading = 0

# Simulation
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(currentdir, "../gym")
os.sys.path.insert(0, parentdir)
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)
useRealTimeSim = 1
p.setRealTimeSimulation(useRealTimeSim)
p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), basePosition=[0, 0, 0])

# Car
car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
inactive_wheels = [5, 7]
wheels = [2, 3]
for wheel in inactive_wheels:
  p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
steering = [4, 6]

BARRIER = [] # List to hold generated barriers
BARRIER_IDS = {} # Dictionary to hold generated barriers along with their IDs
MAX_OBSTACLES = 40

# Cylinder shape for the barriers
cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.5)

# Function to check if a new barrier is too close to existing ones
def too_close(new_barrier, existing_barriers, min_distance=1):
    for barrier in existing_barriers:
        dist = math.sqrt((new_barrier[0]-barrier[0])**2 + (new_barrier[1]-barrier[1])**2)
        if dist < min_distance:
            return True
    return False

def spawnObstaclesAroundCar():
    new_barrier_polar = [random.uniform(2, 9), random.uniform(-0.4, 0.4)]
    new_barrier_cartesian = [new_barrier_polar[0] * math.cos(new_barrier_polar[1]) + x, 
                            new_barrier_polar[0] * math.sin(new_barrier_polar[1]) + y]
    # global BARRIER
    if not too_close(new_barrier_cartesian, BARRIER):
        BARRIER.append(new_barrier_cartesian) 

        # Generate the barriers in the simulation
        obstacle_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=cylinder_shape,
            basePosition=[new_barrier_cartesian[0], new_barrier_cartesian[1], 0.3],
        )

        # Add the obstacle id to the dictionary
        BARRIER_IDS[tuple(new_barrier_cartesian)] = obstacle_id

        if len(BARRIER) > MAX_OBSTACLES:
            # Remove the oldest obstacle
            oldest_obstacle = BARRIER.pop(0)
            p.removeBody(BARRIER_IDS[tuple(oldest_obstacle)])
            del BARRIER_IDS[tuple(oldest_obstacle)]

def updatePosition():
    global x, y, z, heading
    pos, hquat = p.getBasePositionAndOrientation(car)
    heading = p.getEulerFromQuaternion(hquat)[2]
    x = pos[0]
    y = pos[1]
    z = pos[2]

# Move to Point function
def moveTo(targetX, targetY):
  updatePosition()
  distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
  theta = math.atan2((targetY - y), (targetX - x))

  while distance > 1:
    updatePosition()
    spawnObstaclesAroundCar()
    spawnObstaclesAroundCar()
    spawnObstaclesAroundCar()

    distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
    theta = math.atan2((targetY - y), (targetX - x))
    maxForce = 20
    targetVelocity = 10*distance

    # velocity cap
    if targetVelocity > 15:
       targetVelocity = 15

    # make sure steering angle doesnt exceed -pi or pi
    steeringAngle = theta - heading
    if abs(steeringAngle) > (math.pi):
       steeringAngle = heading - theta
    else:
       steeringAngle = theta - heading

    # depth camera stuff
    view_matrix_car = p.computeViewMatrix(
       cameraEyePosition = [x + 0.5*math.cos(heading), y + 0.5*math.sin(heading), z + 0.1],
       cameraTargetPosition = [x + 2*math.cos(heading), y + 2*math.sin(heading), z + 0.05],
       cameraUpVector = [0, 0, 1]
    )
    projection_matrix_car = p.computeProjectionMatrixFOV(
      fov=80,  # field of view
      aspect=1.0,  # aspect ratio
      nearVal=0.1,  # near clipping plane
      farVal=100.0  # far clipping plane
    )

    img_arr = p.getCameraImage(256, 256, viewMatrix=view_matrix_car, projectionMatrix=projection_matrix_car, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_buf = np.reshape(img_arr[3], (256, 256))
    depth_img = 1 - np.array(depth_buf)
  
    firsthalf = depth_img[0][:128]
    totalfirsthalf = sum(firsthalf)
    secondhalf = depth_img[0][128:]
    totalsecondhalf = sum(secondhalf)
    middle = depth_img[0][54:202]
    totalmiddle = sum(middle)

    difference = abs(totalfirsthalf - totalsecondhalf)

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
    
    p.resetDebugVisualizerCamera(
      cameraDistance = 10,
      cameraYaw = -90, 
      cameraPitch = -45,
      cameraTargetPosition = [x + 5, 0, z],
      physicsClientId=0
    )

    for wheel in wheels:
        p.setJointMotorControl2(car,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=targetVelocity,
                            force=maxForce)
    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)
    if (useRealTimeSim == 0):
        p.stepSimulation()
    time.sleep(0.001)

moveTo(100, 0)

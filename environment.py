#%%
import pybullet as p
from math import pi, sin, cos, tan, radians, hypot, sqrt
from time import sleep
import numpy as np
from skimage.transform import resize
import random
import colorsys

from utils import relative_to



image_size = 28
max_wheel_speed = 25
angular_scaler = .5
steps_per_step = 30
num_walls = 0



physicsClient = p.connect(p.GUI)
start_cam = (1, 90, -89, (0, 0, 5))
p.resetDebugVisualizerCamera(1, 90, -89, (0, 0, 10), physicsClientId = physicsClient)
p.setAdditionalSearchPath("pybullet_data")
p.setGravity(0, 0, 0, physicsClientId = physicsClient)
p.setTimeStep(1, physicsClientId=physicsClient)  # More accurate time step
p.setPhysicsEngineParameter(numSolverIterations=1, numSubSteps=4, physicsClientId=physicsClient)
p.setTimeStep(1/240)


    
class Orb:
    
    def __init__(self, pos, color):
        self.pos = pos
        self.index = p.loadURDF(
            "pybullet_data/wall/wall.urdf", 
            pos, 
            useFixedBase=True, 
            globalScaling = 1, 
            physicsClientId=physicsClient)
        
        self.color = color
        p.changeVisualShape(
            self.index, 
            -1, 
            rgbaColor = color, 
            physicsClientId = physicsClient)
        
    def change_pos_color(self, pos, color):
        self.pos = pos
        p.resetBasePositionAndOrientation(
            self.index, 
            pos, 
            (0, 0, 0, 1), 
            physicsClientId = physicsClient)

        self.color = color
        p.changeVisualShape(
            self.index, 
            -1, 
            rgbaColor = color, 
            physicsClientId = physicsClient)
        
        
        
class Wall:
    
    def __init__(self, pos):
        self.pos = pos 
        self.index = p.loadURDF(
            "pybullet_data/wall/wall.urdf", 
            pos, 
            useFixedBase=True, 
            globalScaling = 1, 
            physicsClientId=physicsClient)
        p.changeVisualShape(
            self.index, 
            -1, 
            rgbaColor = [0, 0, 0, 1], 
            physicsClientId = physicsClient) 
        
    def change_pos(self, pos):
        self.pos = pos
        p.resetBasePositionAndOrientation(
            self.index, 
            pos, 
            (0, 0, 0, 1), 
            physicsClientId = physicsClient)
        
        
        
def make_positions(positions, n):
    min_distance = 2
    max_distance = 5
    start_length = len(positions)
    while(len(positions) < start_length + n):
        # I would rather this be circular.
        x = random.uniform(max_distance, -max_distance)
        y = random.uniform(max_distance, -max_distance)
        if all(hypot(x - pos[0], y - pos[1]) >= min_distance for pos in positions):
            positions.append([x, y])
    return(positions[-n:])



class Environment():
    def __init__(self):
        
        self.robot_start_position = [0, 0, .5]
        self.robot_start_orientation = [pi/2, 0, 0]
        self.wheel_speeds = [0, 0]
        
        # Make robot.                 
        self.robot_index = p.loadURDF(
            "pybullet_data/duck/duck.urdf", 
            self.robot_start_position,
            p.getQuaternionFromEuler(self.robot_start_orientation), 
            useFixedBase=False, 
            globalScaling = 10, 
            physicsClientId = physicsClient)
        p.changeVisualShape(
            self.robot_index, 
            -1, 
            rgbaColor = (1, 0, 0, 1), 
            physicsClientId = physicsClient)
        
        # Make objects. 
        self.orbs = []
        for i in range(3):
            self.orbs.append(Orb(pos = [i+1, i+1, 1], color = [0, 0, 0, 1]))
        self.correct_orb_index = self.orbs[0].index
            
        # Make walls. 
        self.walls = []
        for i in range(num_walls):
            self.walls.append(Wall(pos = [-i-1, -i-1, 1]))
                            

                                
    def begin(self):
        self.set_pos(self.robot_start_position)
        self.set_orn(self.robot_start_orientation)
        self.set_wheel_speeds()    
        
        A = [-3, 0] 
        B = [3/2, 3*sqrt(3)/2] 
        C = [3/2, -3*sqrt(3)/2]
        
        orb_positions = [A, B, C]
        
        # Random rotation angle
        theta = random.uniform(0, 2*pi)
        
        def rotate(point, angle):
            x, y = point
            return (
                [x*cos(angle) - y*sin(angle),
                x*sin(angle) + y*cos(angle)]
            )
        
        # Rotate all points
        orb_positions = [rotate(p, theta) for p in orb_positions]
        
        positions = [[0, 0]]
        #orb_positions = make_positions(positions, len(self.orbs))
        wall_positions = make_positions(positions + orb_positions, len(self.walls))
                    
        h = random.random() 
        s = random.uniform(0.5, 1.0) 
        l = random.uniform(0.4, 0.6)
        color_1 = list(colorsys.hls_to_rgb(h, l, s))
        color_1.append(1)
        h = (h + 0.5) % 1.0
        color_2 = list(colorsys.hls_to_rgb(h, l, s))
        color_2.append(1)
                
        for i in range(len(self.orbs)):
            self.orbs[i].change_pos_color(orb_positions[i] + [1], color_1 if i == 0 else color_2)
            
        for i in range(len(self.walls)):
            self.walls[i].change_pos(wall_positions[i] + [1])
                        
            
            
    def step(self, left_wheel_speed, right_wheel_speed, sleep_time = 0):
                
        left_wheel_speed *= -1
        right_wheel_speed *= -1
        
        pos_orn_spe = self.get_pos_orn_spe(self.robot_index)
        left_wheel_speed_start, right_wheel_speed_start = self.wheel_speeds[0], self.wheel_speeds[1]
        left_wheel_speed_end = relative_to(left_wheel_speed, -max_wheel_speed, max_wheel_speed)
        right_wheel_speed_end = relative_to(right_wheel_speed, -max_wheel_speed, max_wheel_speed)
        
        change_in_left_wheel = left_wheel_speed_end - left_wheel_speed_start
        change_in_left_wheel_per_step = change_in_left_wheel / steps_per_step
        change_in_right_wheel = right_wheel_speed_end - right_wheel_speed_start
        change_in_right_wheel_per_step = change_in_right_wheel / steps_per_step   
                
        for step in range(steps_per_step):   
            if(sleep_time != 0):
                sleep(sleep_time)
            left_wheel_step = left_wheel_speed_start + change_in_left_wheel_per_step * (step + 1)
            right_wheel_step = right_wheel_speed_start + change_in_right_wheel_per_step * (step + 1)
            self.set_wheel_speeds([left_wheel_step, right_wheel_step]) 
                        
            p.stepSimulation(physicsClientId = physicsClient)
            
            pos_orn_spe = self.get_pos_orn_spe(self.robot_index)
            orn = self.robot_start_orientation
            orn = [orn[0], orn[1], pos_orn_spe["yaw"]]
            self.set_orn(orn)
            
        return(self.positions_for_plot())
            
            
            
    # Reward
    def reward(self):
        reward = 0
        for key, value in self.touching_orbs().items():
            if(value):
                if(key == self.correct_orb_index):
                    reward += 0
                else:
                    reward += 0
        return(reward)
            
        
            
    # Functions for agent positions/angles
    def get_pos_orn_spe(self, index):
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId = physicsClient)
        roll, pitch, yaw = p.getEulerFromQuaternion(ors, physicsClientId = physicsClient)
        
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_index, physicsClientId=physicsClient)
        vx, vy, _ = linear_velocity  
        _, _, angular_velocity = angular_velocity  
            
        return(
            {"pos" : pos,
             "roll" : roll,
             "pitch" : pitch,
             "yaw" : yaw,
             "linear_velocity" : linear_velocity,
             "angular_velocity" : angular_velocity})
    
    
    
    def set_pos(self, pos):
        yaw = self.get_pos_orn_spe(self.robot_index)["yaw"]
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId = physicsClient)
        
    def set_orn(self, orn):
        pos_orn_spe = self.get_pos_orn_spe(self.robot_index)
        if(len(orn) == 3):
            orn = p.getQuaternionFromEuler(orn)
        p.resetBasePositionAndOrientation(self.robot_index, pos_orn_spe["pos"], orn, physicsClientId = physicsClient)
        p.resetBaseVelocity(
            self.robot_index, 
            linearVelocity=self.wheel_speeds + [0], 
            angularVelocity=pos_orn_spe["angular_velocity"], 
            physicsClientId = physicsClient)
        
    def set_wheel_speeds(self, wheel_speeds = [0, 0]):
        self.wheel_speeds = wheel_speeds 
        linear_velocity = sum(wheel_speeds) / 2
        yaw = self.get_pos_orn_spe(self.robot_index)["yaw"]
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (wheel_speeds[0] - wheel_speeds[1]) * angular_scaler
        p.resetBaseVelocity(
            self.robot_index, 
            linearVelocity=[x, y, 0], 
            angularVelocity=[0, 0, angular_velocity], 
            physicsClientId = physicsClient)
        
    def touching_orbs(self):
        touching = {}
        for orb in self.orbs:
            touching_this = bool(p.getContactPoints(
                bodyA=self.robot_index, bodyB=orb.index, physicsClientId = physicsClient))
            touching[orb.index] = True if touching_this else False
        return(touching)
        
        
        
    def photo_for_agent(self):
        fov_x_deg = 90
        fov_y_deg = 90
        fov_x_rad = radians(fov_x_deg)
        fov_y_rad = radians(fov_y_deg)
        
        near = 0.4
        far = 20
        
        right = near * tan(fov_x_rad / 2)
        left = -right
        top = near * tan(fov_y_rad / 2)
        bottom = -top

        pos_orn_spe = self.get_pos_orn_spe(self.robot_index)
        pos, yaw = pos_orn_spe["pos"], pos_orn_spe["yaw"]
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0] - x*.1, pos[1] - y*.1, 1], 
            cameraTargetPosition = [pos[0] - x*2, pos[1] - y*2, 1],    
            cameraUpVector = [0, 0, 1], physicsClientId = physicsClient)
        proj_matrix = proj_matrix = p.computeProjectionMatrix(
            left, 
            right, 
            bottom, top, near, far)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=image_size * 2, 
            height=image_size * 2,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = physicsClient)
        
        if(type(rgba) == np.ndarray):
            pass
        else:
            rgba = np.array(rgba).reshape(32, 32, 4)
            depth = np.array(depth).reshape(32, 32)
            
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d - d.min())/(d.max()-d.min())
        vision = np.concatenate([rgb, d], axis = -1)
        vision = resize(vision, (image_size, image_size, 4))
        return(vision)
    
    
    
    def positions_for_plot(self):
        pos = {}
        pos["robot"] = self.get_pos_orn_spe(self.robot_index)
        pos["orbs"] = []
        pos["walls"] = []
        for orb in self.orbs:
            pos["orbs"].append(orb.pos)
        for wall in self.walls:
            pos["walls"].append(wall.pos)
        return(pos)
    
        
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = Environment()
    env.begin()
    
    while True:
        env.step(.5, 1, sleep_time = .01)
        vision = env.photo_for_agent()
        plt.imshow(vision[:,:,:3])
        plt.show()
        print(env.reward())
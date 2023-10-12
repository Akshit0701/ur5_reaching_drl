#!/usr/bin/env python

"""UR5 Environment"""

# Gazebo Imports
import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
import control_msgs.msg
import actionlib
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Empty

import numpy as np
import gymnasium as gym
import sys
import torch
import time

# Robotics toolbox -python imports for kinematics and dynamics of ur5
import roboticstoolbox as rtb
from spatialmath import SE3

class ROBO_ENV():

    def __init__(self):

        rospy.init_node('ROBO_ENV', anonymous = True) # Initializing node

        self.jointstate = JointState()
        self.modelstate = ModelState()
        self.q_cmd = JointTrajectory()
        self.q_cmd.joint_names = ['ur5_arm_shoulder_pan_joint', 'ur5_arm_shoulder_lift_joint', 'ur5_arm_elbow_joint', 'ur5_arm_wrist_1_joint', 'ur5_arm_wrist_2_joint', 'ur5_arm_wrist_3_joint']
        self.point = JointTrajectoryPoint()

        self.cube_name = 'cube1'
        self.cube_relative_entity_name = 'link'
        self.robot = rtb.models.UR5() # Load UR5

        # Gazebo Services
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        # Publisher and Subscriber 
        self.ur_cmd = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size = 1)
        self.ur_jointstate = rospy.Subscriber('/joint_states', JointState, self.ur5_joint_callback)
        self.gripper_client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', control_msgs.msg.GripperCommandAction) # 0.0 = open & 0.8 = close
        self.ft_sensor = rospy.Subscriber('/ft_sensor/raw', WrenchStamped, self.ft_sensor_callback)
        self.goal = control_msgs.msg.GripperCommandGoal()
        
        # Limits of end-effector position
        self.max = np.array([0.60, 0.22, 0.40])#30])
        self.min = np.array([0.29, -0.22, 0.188])#188])
        self.max_x = torch.tensor(self.max, dtype = torch.float32, device = torch.device("cpu"))
        self.min_x = torch.tensor(self.min, dtype = torch.float32, device = torch.device("cpu"))

        # Action space = [x_direction,y_direction,z_direction] (task space)
        self.action_space = gym.spaces.Box(low = np.array([-3,-4,-2]), high = np.array([3,4,2]), dtype= np.float32)
        # self.action_space = gym.spaces.Box(low = np.array([-15.5, -22, -5.6]), high = np.array([15.5, 22, 5.6]), dtype= np.float32)
        # self.action_space = gym.spaces.Box(low = np.array([29, -22, 18.8]), high = np.array([60, 22, 30]), dtype= np.float32)
        self.max_action = self.action_space.high
        self.min_action = self.action_space.low

        # Observation Space = [x,y,z,cube.x,cube.y,cube.z]
        self.observation_space = gym.spaces.Box(low = np.array([30, -25, 20, 35, -15, 0]), high = np.array([70, 25, 35, 45, 0, 5]), dtype=np.float32)

        self.reward = 0
        self.prev_reward = 0
        self.prev_distToGoal = 0
        self.distToGoal = 0
        self.done_counter = 0

    
    def ur5_joint_callback(self, data):

        self.jointstate = data

    def ft_sensor_callback(self,data):

        self.ft_data = data

    def get_observation(self):

        # Cube Coordinates
        self.cube_coord = self.model_coordinates(self.cube_name, self.cube_relative_entity_name)
        
        # UR5 XYZ
        self.q0 = self.jointstate.position
        self.Te = np.array(self.robot.fkine(self.q0))
        self.x0 = np.array([self.Te[0][3],self.Te[1][3],self.Te[2][3]])
        # self.x_cube = np.array([self.cube_coord.position.x,self.cube_coord.position.y, self.cube_coord.position.z])
        # self.x_cube = np.array([self.cube_coord.pose.position.x, self.cube_coord.pose.position.y, self.cube_coord.pose.position.z])
        self.x0[2] -= 0.17
        #Creating observation array
        # self.x_goal = np.array([40, -10, 2.5])
        self.x_goal = np.array([100*self.cube_x, 100*self.cube_y, 2.5])
        self.x0 = 100*self.x0
        self.obs = np.array([])
        self.obs = np.append(self.obs, self.x0)
        self.obs = np.append(self.obs, self.x_goal)
        # self.obs = np.append(self.obs, self.x_cube)

        return self.obs


    def reset(self):

        self.q_cmd = JointTrajectory()
        self.q_cmd.joint_names = ['ur5_arm_shoulder_pan_joint', 'ur5_arm_shoulder_lift_joint', 'ur5_arm_elbow_joint', 'ur5_arm_wrist_1_joint', 'ur5_arm_wrist_2_joint', 'ur5_arm_wrist_3_joint']
        self.point = JointTrajectoryPoint()
        self.unpause()
        self.q = [0.0,-1.57,1.57,-1.57,-1.57,1.57]
        # UR5 reset position
        # self.point.positions = [0.0,-1.57,1.57,-1.57,-1.57,1.57]
        self.q_dot_cmd = [0.1,0.1,0.1,0.1,0.1,0.1]
        self.og_Te = np.array(self.robot.fkine(np.array([0.0,-1.57,1.57,-1.57,-1.57,1.57])))
        
        # Randomize UR5 gripper x and y location
        self.ur_x = np.random.uniform(0.30,0.59)
        self.ur_y = np.random.uniform(-0.15,0.15)
        self.og_Te[0][3] = self.ur_x
        self.og_Te[1][3] = self.ur_y
        self.sol = self.robot.ikine_LM(SE3(self.og_Te), q0 = self.q)
        self.point.positions = self.sol.q

        # Publish UR5 velocity and position
        self.point.velocities = -3*self.q_dot_cmd
        self.point.time_from_start = rospy.Duration(1)
        self.q_cmd.points.append(self.point)
        self.ur_cmd.publish(self.q_cmd)
        time.sleep(0.5)


        # Publish gripper as open and set gripper_status = 0
        self.gripper_status = 0
        self.gripper_client.wait_for_server()
        self.goal.command.position = self.gripper_status # from 0.0 (open) to 0.8 (close)
        self.goal.command.max_effort = -1.0 # Do not limit the effort
        self.gripper_client.send_goal(self.goal)
        self.gripper_client.wait_for_result()

        # Randomize x and y location of cube
        self.cube_x = np.random.uniform(0.3,0.5)
        self.cube_y = np.random.uniform(-0.15,0.15)
        print("\ncube_x = ", self.cube_x)
        print("\ncube_y = ", self.cube_y)
        # Cube reset position
        self.modelstate.model_name = 'cube1'
        self.modelstate.pose.position.x = self.cube_x #0.4
        self.modelstate.pose.position.y = self.cube_y #-0.1
        self.modelstate.pose.position.z = 0.6
        self.modelstate.pose.orientation.x = 0
        self.modelstate.pose.orientation.y = 0
        self.modelstate.pose.orientation.z = 0
        self.modelstate.pose.orientation.w = 0
        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            self.resp = self.set_state(self.modelstate)

        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

        self.obs = self.get_observation()
        self.reward = 0
        self.prev_reward = 0
        self.stage = 0
        self.pause()

        return self.obs

    def calculate_reward(self, new_obs, prev_obs):

        self.reward = 0
        self.new_obs = new_obs
        self.new_x0 = self.new_obs[0:3]
        # self.x_goal = np.array([40, -10, 2.5])
        self.x_goal = self.new_obs[-3:]

        self.prev_obs = prev_obs
        self.prev_x0 = self.prev_obs[0:3]

        self.distToGoal = np.linalg.norm(self.x_goal - self.new_x0)
        print("\nDist to goal = ", self.distToGoal)
        self.prev_distToGoal = np.linalg.norm(self.x_goal - self.prev_x0)

        self.distdifference = self.distToGoal - self.prev_distToGoal
        print("\nDist difference = ", self.distdifference)
        # self.reward = (100/self.distToGoal)#**2
        self.reward = -self.distToGoal

        # Reward based on lunar lander
        # self.x_dist = -0.1*np.sqrt(self.new_x0[0]*self.new_x0[0] + self.x_goal[0]*self.x_goal[0])
        # self.y_dist = -0.1*np.sqrt(self.new_x0[1]*self.new_x0[1] + self.x_goal[1]*self.x_goal[1])
        # self.z_dist = -0.1*np.sqrt(self.new_x0[2]*self.new_x0[2] + self.x_goal[2]*self.x_goal[2])

        # self.reward += self.x_dist + self.y_dist + self.z_dist
        
        # if self.distdifference < 0:
        #     self.reward += 100*(-self.distdifference/self.distToGoal)
        # else:
        #     self.reward -= 100*(-self.distdifference/self.distToGoal)

        # if self.prev_reward is not None:
        #     self.reward += self.prev_reward

        # if self.distdifference < 0.0:
        #     self.reward += (-self.distdifference)
        # else:
        #     self.reward += (self.distdifference) #distdifference is negative so neg reward

        if self.distToGoal <= 2.5:#3.5:
            self.reward += 2000#1000
            if np.linalg.norm(self.new_x0[0] - self.x_goal[0]) < 1:
                self.reward += 200
            if np.linalg.norm(self.new_x0[1] - self.x_goal[1]) < 1:
                self.reward += 200
            self.done = True
            self.done_counter +=1
            print("\ndone_counter =", self.done_counter)
            # time.sleep(0.1)
        else:
            self.done = False
            # self.reward -= 20

        # self.prev_reward = self.reward

        print("\nReward: ", self.reward)

        return self.reward, self.done

    def step(self,action):

        self.pause()
        self.q_cmd = JointTrajectory()
        self.q_cmd.joint_names = ['ur5_arm_shoulder_pan_joint', 'ur5_arm_shoulder_lift_joint', 'ur5_arm_elbow_joint', 'ur5_arm_wrist_1_joint', 'ur5_arm_wrist_2_joint', 'ur5_arm_wrist_3_joint']
        self.point = JointTrajectoryPoint()

        # self.modelstate.model_name = 'cube1'
        # self.modelstate.pose.position.x = 0.4
        # self.modelstate.pose.position.y = 0.0
        # self.modelstate.pose.position.z = 0.6
        # self.modelstate.pose.orientation.x = 0
        # self.modelstate.pose.orientation.y = 0
        # self.modelstate.pose.orientation.z = 0
        # self.modelstate.pose.orientation.w = 0
        # rospy.wait_for_service('/gazebo/set_model_state')

        # try:
        #     self.resp = self.set_state(self.modelstate)

        # except rospy.ServiceException as e:
        #     print ("Service call failed: %s" % e)

        # time.sleep(0.1)

        # self.x = np.array([0.445, 0.0, 0.244])

        self.action = action
        # print("\nNetwork output action = ", action)
        # self.action = self.min_action + (np.array(action) + 1.0)*0.5*(self.max_action-self.min_action)
        # self.action = self.max_action*self.action#np.array(action)
        # self.action = np.clip(self.action, self.min_action, self.max_action)
        # print("\nNormalized Action: ", self.action)

        self.x = np.array(self.action)*0.01
        print("\nx: ", self.x)

        self.prev_obs = self.get_observation()

        self.Te = np.array(self.robot.fkine(self.q0))
        
        self.x[0] += self.Te[0][3]
        self.x[1] += self.Te[1][3]
        self.x[2] += self.Te[2][3]

        # self.x_tensor = torch.tensor(self.x, dtype = torch.float32, device = torch.device("cpu"))

        self.x_cliped = np.clip(self.x, self.min_x, self.max_x)
        print("\nAction: ", self.x_cliped)
        
        self.og_Te[0][3] = self.x_cliped[0]
        self.og_Te[1][3] = self.x_cliped[1]
        self.og_Te[2][3] = self.x_cliped[2]

        # Calculate joint positions
        self.sol = self.robot.ikine_LM(SE3(self.og_Te), q0 = self.q0)
        self.point.positions = self.sol.q
        # print(self.sol.q)
        self.q_dot_cmd = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
        # time.sleep(0.3)
        self.unpause()
        # Publish UR5 velocity and position
        self.point.velocities = self.q_dot_cmd
        self.point.time_from_start = rospy.Duration(1.2)
        self.q_cmd.points.append(self.point)
        # print("q_cmd: ", self.q_cmd)
        self.ur_cmd.publish(self.q_cmd)
        time.sleep(0.5)
        self.pause()

        self.new_obs = self.get_observation()
        self.reward, self.done = self.calculate_reward(self.new_obs, self.prev_obs)

        self.info = None
        
        return self.new_obs, self.reward, self.done, self.info
import random
import heapq
import math
from math import atan2, sin, cos, tan, pi

import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.constants import value

from env_formation.obstacle import obstacle


class leader_agent:
    def __init__ (self, pos, vel):
        self.pos = pos
        self.xy_vel = vel
        self.orientation = 0
        self.vel = [0, 0]
        self.done = False
        self.target = False
    
    def take_action(self, linear_vel, steer_vel, dt=0.1):
        L = 2.5
        new_orientation = self.orientation + (linear_vel / L) * tan(steer_vel) * dt
        print(f"角度变化量:{(linear_vel / L) * tan(steer_vel) * dt}")
        new_orientation = new_orientation % (2 * pi)  # 规范化到 [0, 2pi]

        dx = linear_vel * dt * cos(new_orientation)
        dy = linear_vel * dt * sin(new_orientation)
        self.pos[0] += dx
        self.pos[1] += dy
        self.orientation = new_orientation
        self.xy_vel = [linear_vel*cos(new_orientation), linear_vel*sin(new_orientation)]
        self.vel = [linear_vel, steer_vel]
    
    def set_position(self, x, y):
        self.pos = [x, y]
    
    def position(self):
        return np.array(self.pos)



class follower_agent:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.obs_done = False
        self.side_done = False
        self.uav_done = False
        self.formation_done = False
        self.done = False
        self.target = False
        self.observation = np.array([])

    
    def take_action(self, vx, vy, dt=0.1):
        self.pos[0] = self.pos[0] + vx*dt
        self.pos[1] = self.pos[1] + vy*dt

        self.vel = [vx, vy]

    def set_position(self, x, y):
        self.pos = [x, y]
    
    def position(self):
        return np.array(self.pos)
    
    def set_vel(self, x, y):
        self.vel[0] = x
        self.vel[1] = y

class APF_env:
    def __init__(self, ):
        self.leader_agent = leader_agent(pos=[20, 20], vel=[0, 0])
        self.follower_uav_num = 3
        self.follower_uavs = {}
        self.width = 100
        self.height = 100
        self.num_obstacles = 15
        self.obs_radius = 2
        self.target_radius = 4
        self.agent_radius = 1
        self.obs_delta =10
        self.safe_theta = 8
        self.fig = None
        self.ax = None
        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"] = follower_agent(pos=[0, 0], vel=[0, 0])

        self.formation_pos = [
            [0, 2*3],
            [-1.732*3, -1*3],
            [1.732*3, -1*3]
        ]

        self.fix_position =  [
            (25, 25),
            (50, 50),
            (75, 75),
            (75, 25),
            (25, 75),
            (25, 50),
            (50, 25),
            (50, 75),
            (75, 50)
        ]

        self.obstacles = {}
        for i in range(len(self.fix_position)):
            pos_x, pos_y = self.fix_position[i]
            self.obstacles[f"obstacle_{i}"] = obstacle(
                radius=self.obs_radius,
                pos_x=pos_x,
                pos_y=pos_y,
                safe_theta=self.safe_theta
            )

        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"] = obstacle(
                radius=self.obs_radius,
                pos_x=np.random.rand() * self.width * 0.7 + self.width * 0.15,
                pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15,
                safe_theta=self.safe_theta
            )
        self.leader_agent_trajectory = []
        self.follower_agent_trajectory = []
        self.follower_agent_trajectory = [[] for _ in range(self.follower_uav_num)]
        self.leader_target_pos = np.array([0, 0])
        self._check_obs()
        self.reset()
    
    def _check_obs(self):
        """ 确保障碍物不重复 """
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for i, obs in enumerate(random_obstacles):
            key = obstacles_keys[9 + i]
            is_position_valid = False

            while not is_position_valid:
                is_position_valid = True

                # 仅检查与之前的随机障碍物的距离
                for j in range(i):
                    obs2 = random_obstacles[j]
                    dis = np.linalg.norm(self.obstacles[key].position() - obs2.position())
                    if dis < 2 * self.obs_radius + self.agent_radius + self.safe_theta:
                        is_position_valid = False
                        break

                # 检查与固定障碍物的距离
                for fixed_obs in fixed_obstacles:
                    dis_fixed = np.linalg.norm(self.obstacles[key].position() - fixed_obs.position())
                    if dis_fixed < 2 * self.obs_radius + self.agent_radius + self.safe_theta:
                        is_position_valid = False
                        break

                # 检查与目标位置的距离
                dis2 = np.linalg.norm(np.array(self.leader_target_pos) - self.obstacles[key].position())
                if dis2 < self.obs_radius + self.target_radius + self.agent_radius + self.safe_theta:
                    is_position_valid = False

                # 如果位置无效，则重新生成随机位置
                if not is_position_valid:
                    self.obstacles[key].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
                    self.obstacles[key].pos_y = np.random.rand() * self.height * 0.7 + self.height * 0.15
            
    def _check_obs_agent(self, agent):
        # obstacles_keys = list(self.obstacles.keys())
        # obstacles_list = list(self.obstacles.values())

        # # 假设前9个障碍物是固定的
        # fixed_obstacles = obstacles_list[:9]
        # random_obstacles = obstacles_list[9:]
        for obs in self.obstacles.values():
            dis = np.linalg.norm(obs.position() - agent.position())
            if dis <= self.obs_radius + self.agent_radius + self.safe_theta/2:
                return True
        for i in range(self.follower_uav_num):
            for obs in self.obstacles.values():
                dis = np.linalg.norm(obs.position() - self.follower_uavs[f"follower_{i}"].position())
                if dis <= self.obs_radius + self.agent_radius + self.safe_theta/2:
                    return True  
        return False
    
    def _check_obs_target(self, target_pos):
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for obs in fixed_obstacles:
            dis = np.linalg.norm(obs.position() - np.array(target_pos))
            if dis < self.obs_radius + self.target_radius + self.safe_theta/2:
                return True
        return False
    
    def _check_fix_obs_agent(self, leader_pos):
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for obs in fixed_obstacles:
            if np.linalg.norm(obs.position() - np.array(leader_pos)) < self.agent_radius + self.obs_radius +self.safe_theta/4:
                return True
            
            for i in range(self.follower_uav_num):
                if np.linalg.norm(obs.position() - np.array(self.follower_uavs[f"follower_{i}"].position())) < self.agent_radius +self.obs_radius + self.safe_theta/4:
                    return True
                
            return False

    
    def reset(self):
        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
            self.obstacles[f"obstacle_{i}"].pos_y=np.random.rand() * self.height * 0.7 + self.height * 0.15

        self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]
        # self.leader_target_pos = [85 + np.random.rand() * 5 , 85 + np.random.rand() * 5]
        self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
        # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)
        
        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])

        # 确保开始不与固定障碍物碰撞
        flag0 = self._check_fix_obs_agent(self.leader_agent.pos)
        while flag0:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)
            
            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])
            flag0 = self._check_fix_obs_agent(self.leader_agent.pos)

        
        
        # 确保目标不与固定障碍物碰撞
        flag2 = self._check_obs_target(self.leader_target_pos)
        while flag2:
            self.leader_target_pos = [self.width*0.1+np.random.rand()*self.width*0.8, self.height*0.1+np.random.rand()*self.height*0.8]
            # self.leader_target_pos = [85 + np.random.rand() * 5 , 85 + np.random.rand() * 5]
            
            flag2 = self._check_obs_target(self.leader_target_pos)

        self._check_obs()
        
        flag1 = self._check_obs_agent(self.leader_agent)
        while  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos)) < self.agent_radius + self.target_radius + self.safe_theta*3 or flag1:
            self.leader_agent.set_position(self.width*0.10+np.random.rand()*self.width*0.8, self.height*0.10+np.random.rand()*self.height*0.8)
            # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)
            
            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_agent.pos[0] + self.formation_pos[i][0], self.leader_agent.pos[1] + self.formation_pos[i][1])
            flag1 = self._check_obs_agent(self.leader_agent)

        target_distance =  np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_agent.pos))
        
        self.leader_agent.orientation = np.random.rand()*2*np.pi

        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_vel(0, 0)
            self.follower_uavs[f"follower_{i}"].target = False
            self.follower_uavs[f"follower_{i}"].done = False
            self.follower_uavs[f"follower_{i}"].obs_done = False
            self.follower_uavs[f"follower_{i}"].side_done = False
            self.follower_uavs[f"follower_{i}"].uav_done = False
            self.follower_uavs[f"follower_{i}"].formation_done = False

        self.leader_agent.done = False
        self.leader_agent.target = False

    def caculate_force(self, target_pos, self_pos, obs_num, is_follower=False, self_id=0):
        # 跟目标的距离和角度
        force_list = []
        target_dis, target_angle = APF_env.calculate_relative_distance_and_angle(self_pos, target_pos)
        force_list.append([target_dis,  target_angle])
        # force_list.append([target_dis,  (target_angle + np.pi)%(2*np.pi)])

        # for obs_id, obs in self.obstacles.items():
        #     dis_, angle_ = APF_env.calculate_relative_distance_and_angle(self_pos,[obs.pos_x, obs.pos_y])
        #     dis = dis_ - self.obs_radius - self.agent_radius
        #     angle = (angle_ + np.pi) % (2*np.pi)
        #     if dis <= self.obs_delta*1.5 - self.obs_radius - self.agent_radius:     
        #         force_list.append([dis, angle])
        
        if is_follower:
            for j in range (self.follower_uav_num):
                if j != self_id:
                        dis_, angle_ = APF_env.calculate_relative_distance_and_angle(self_pos,self.follower_uavs[f"follower_{j}"].position())
                        dis = dis - 2*self.agent_radius
                        angle = (angle_ + np.pi) % 2*np.pi
                        force_list.append([dis, angle])
        
        return APF_env.compute_resultant_force(force_list)
        # return [target_dis, target_angle]


    def render(self, display_time = 0.1, force = None):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10,10), dpi=100)
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
        
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        leader_agent = patches.Circle(self.leader_agent.pos, self.agent_radius, color='purple', fill=True)
        self.ax.add_patch(leader_agent)
        arrow_length = self.agent_radius * 1  # Adjust length as needed
        arrow_dx = arrow_length * np.cos(self.leader_agent.orientation)
        arrow_dy = arrow_length * np.sin(self.leader_agent.orientation)
        arrow = patches.FancyArrow(
            self.leader_agent.pos[0], 
            self.leader_agent.pos[1], 
            arrow_dx, 
            arrow_dy, 
            width=self.agent_radius * 0.25, 
            color='purple'
        )
        self.ax.add_patch(arrow)
        if force != None:
            force_dx = force[0] * np.cos(force[1])
            force_dy = force[0] * np.sin(force[1])
            force_arrow = patches.FancyArrow(
                self.leader_agent.pos[0], 
                self.leader_agent.pos[1], 
                force_dx, 
                force_dy, 
                width=self.agent_radius * 0.25, 
                color='red'
            )

            self.ax.add_patch(force_arrow)
        # 记录智能体当前的位置到轨迹
        self.leader_agent_trajectory.append(self.leader_agent.pos.copy())
        # 绘制智能体的轨迹
        if len(self.leader_agent_trajectory) > 1:
            traj_x, traj_y = zip(*self.leader_agent_trajectory)
            self.ax.plot(traj_x, traj_y, color='blue', linestyle='-', marker='o', markersize=1, label='Trajectory')
        
        colors = ['orange', 'cyan', 'magenta']  # Colors for each follower
        
            

        # 绘制跟随者无人机
        for i in range(self.follower_uav_num):
            uav = patches.Circle(self.follower_uavs[f"follower_{i}"].pos, self.agent_radius, color='orange', fill=True)
            self.ax.add_patch(uav)
            self.follower_agent_trajectory[i].append(self.follower_uavs[f"follower_{i}"].pos.copy())

            # Draw trajectory for this follower
            if len(self.follower_agent_trajectory[i]) > 1:
                traj_x, traj_y = zip(*self.follower_agent_trajectory[i])
                self.ax.plot(traj_x, traj_y, color=colors[i % len(colors)], linestyle='-', marker='o', markersize=1, label=f"Follower {i} Trajectory")

        
        # 绘制障碍物
        obses = [patches.Circle([obs.pos_x, obs.pos_y], self.obs_radius, color='red', fill=True)for obs in self.obstacles.values()]
        for obs_circle in obses:
            self.ax.add_patch(obs_circle)
        # 绘制目标
        target = patches.Circle(self.leader_target_pos, self.target_radius, color='green', fill=True)
        self.ax.add_patch(target)

        plt.pause(display_time)  # 暂停以更新图形
        # plt.show()

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

            self.leader_agent_trajectory = []
            self.follower_agent_trajectory = []
            self.follower_agent_trajectory = [[] for _ in range(self.follower_uav_num)]
        
 
            
        

    @staticmethod
    def calculate_relative_distance_and_angle(pos1, pos2):
        """
        计算两点之间的相对距离和角度

        参数:
        - pos1: 第一个点的位置 (numpy数组或列表 [x, y])
        - pos2: 第二个点的位置 (numpy数组或列表 [x, y])

        返回值:
        - distance: 两点之间的距离
        - angle: 从pos1到pos2的角度（弧度）
        """
        # 计算相对位置向量
        relative_pos = np.array(pos2) - np.array(pos1)
        
        # 计算距离
        distance = np.linalg.norm(relative_pos)
        
        # 计算角度，使用arctan2来得到正确的象限
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        if angle < 0:
            angle = angle + 2*np.pi
        
        
        return distance, angle
     
    @staticmethod
    def compute_resultant_force(forces):
        """
        计算给定一组力的合力。
        
        参数：
        forces (list): 一个列表，包含若干个力，每个力是一个列表，[大小, 角度]，角度单位为弧度。
        
        返回：
        list: 合力的大小和方向，以 [大小, 角度] 的形式返回，角度单位为弧度。
        """
        # 初始化合力的 x 和 y 分量
        resultant_fx = 0
        resultant_fy = 0
        
        # 遍历所有输入的力，计算每个力的 x 和 y 分量
        for force in forces:
            magnitude, angle = force
            fx = magnitude * math.cos(angle)  # 计算 x 分量
            fy = magnitude * math.sin(angle)  # 计算 y 分量
            resultant_fx += fx  # 累加 x 分量
            resultant_fy += fy  # 累加 y 分量
        
        # 计算合力的大小
        resultant_magnitude = math.sqrt(resultant_fx**2 + resultant_fy**2)
        
        # 计算合力的方向（角度），并确保在 [0, 2π] 范围内
        resultant_angle = math.atan2(resultant_fy, resultant_fx)
        if resultant_angle < 0:
            resultant_angle += 2 * math.pi  # 如果是负值，调整到 [0, 2π]
        
        return [resultant_magnitude, resultant_angle]




EPISODE_NUM = 5
STEP_NUM = 3000
env = APF_env()
print(env.leader_target_pos)

env.reset()

for episode_i in range (EPISODE_NUM):
    env.reset()

    for step_i in range (STEP_NUM):
        # print(env.leader_target_pos)
        # print(env.leader_agent.position())
        force = env.caculate_force(target_pos=env.leader_target_pos,
                                   self_pos=env.leader_agent.position(),
                                   obs_num=15)
        
        # === 控制逻辑：根据力来决定速度 ===
        linear_vel = min(force[0] / 25, 1.2) # 直接用合力大小作为线速度

        # 计算合力方向与当前朝向之间的差值
        force_angle = force[1]
        current_orientation = env.leader_agent.orientation
        steer_error = (force_angle - current_orientation) 

        # 假设我们希望前轮转角最大 ±45°（即 ±π/4）
        max_steer_angle = math.radians(45)   # 45° → 0.7854 弧度

        # 对 steer_error 做限幅
        steer_vel = np.clip(steer_error, -max_steer_angle, max_steer_angle)

        # steer_vel = steer_error  # 可以乘一个系数，比如 steer_vel = steer_error * Kp

        # === 执行动作 ===
        print(f"线速度{linear_vel}, 角速度{steer_vel}")
        env.leader_agent.take_action(linear_vel, steer_vel, dt=0.1)

        env.render(display_time = 0.01, force=force)
        
    
    env.render_close()

import random
import heapq
import math
from math import atan2

import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.constants import value

from env_formation.obstacle import obstacle




class APFAgent_leader:
    def __init__(self, list_of_follower_agents, obstacles,leader,goal):
        self.k_rep = None
        self.buffer = None
        self.list_of_agents = list_of_follower_agents
        self.obstacles_list = obstacles
        self.pos = leader.pos
        self.leader=leader
        self.orientation = self.leader.orientation
        self.v = self.leader.vel
        self.goal= goal
        self.display_time = 0.1 # 单轨模型的参数
        self.L = 2.5 # 时间步长
    # 与目标点引力
    def attractive_force(self):
        distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(self.goal))
        self.k_att = 1.0   # 动态调整引力增益

        # 当接近目标点时，增加引力的强度
        if distance_to_goal < 30:
            self.k_att = max(3.0, self.k_att * 1.5)  # 增加引力增益

        attractive_force_ = self.k_att * distance_to_goal
        direction_vector = np.array(self.goal) - np.array(self.pos)
        if distance_to_goal > 0:
            normalized_direction = direction_vector / distance_to_goal
        else:
            normalized_direction = np.array([0, 0])

        attractive_force_x = attractive_force_ * normalized_direction[0]
        attractive_force_y = attractive_force_ * normalized_direction[1]

        return attractive_force_x, attractive_force_y

    # 与障碍斥力
    def repulsive_force(self):
        k_goal = 1

        distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(self.goal))
        self.buffer = 20   # 动态调整斥力缓冲距离
        self.k_rep = 1500
        to_obs = []
        for i,value_ in enumerate(self.obstacles_list.values()):
            dis,theta=value_.calculate_dis_theta(self.pos[0],self.pos[1])
            r_obs = value_.radius
            to_obs.append([dis,theta,r_obs])
        f_rep_x ,f_rep_y = 0, 0
        for dis, theta,r_obs in to_obs:
            # 斥力影响范围
            p0 = self.buffer
            if dis <= p0:
                #f_rep = self.k_rep * ((1 / (dis-self.leader.radius-8) - (1 / (p0-self.leader.radius-8))) * (1 / dis) ** 2) # 原始斥力
                f_rep1 = self.k_rep * (1/dis - 1/p0) * (distance_to_goal**k_goal) * (1/dis)**2 #改进斥力
                f_rep2 = k_goal/2 * self.k_rep * (1/dis - 1/p0)**2 * distance_to_goal**(k_goal-1)
                direction_vector = np.array(self.goal) - np.array(self.pos)
                if distance_to_goal > 0:
                    normalized_direction = direction_vector / distance_to_goal
                else:
                    normalized_direction = np.array([0, 0])  # 如果距离为0，方向向量为0
                f_rep_x += f_rep1 * math.cos(theta) + f_rep2 * normalized_direction[0]
                f_rep_y += f_rep1 * math.sin(theta) + f_rep2 * normalized_direction[1]
                # f_rep_x = f_rep * math.cos(theta)
                # f_rep_y = f_rep * math.sin(theta)
        return f_rep_x, f_rep_y

    # 合力
    def total_force(self):
        a_force_x, a_force_y = self.attractive_force()
        r_force_x, r_force_y = self.repulsive_force()

        # print("tot_fx:", force_x, "tot_fy:", force_y)
        force_x = a_force_x + r_force_x
        force_y = a_force_y + r_force_y

        return force_x, force_y

        # Move the agent towards the goal position based on the total force

    def move_towards(self):
        force_x, force_y = self.total_force()
        # 单轨模型的参数
        L = 2.5  # 前后轮间距，车辆的轴距
        k_w = 30  # 角速度参数
        k_m = 100  # 加速度参数
        dt = self.display_time  # 时间步长
        force = np.linalg.norm(np.array([force_x, force_y])) #合力大小
        angle = np.arctan2(force_y, force_x) - self.orientation # 合力方向与原方向之差
        force_turn = force * math.sin(angle)  # 向心力
        force_run = force * math.cos(angle) # 牵引力
        print("run",force_run)
        print("turn",force_turn)
        a = np.clip(force_run/k_m,-0.2,0.2)
        w = np.clip(force_turn/k_w,-0.5,0.5)  # 正向转向

        self.orientation = self.orientation + np.linalg.norm(self.v)/L * w *np.pi/2* dt
        self.orientation = self.orientation % (2*np.pi)
        last_v = np.linalg.norm(self.v)
        v = np.clip(last_v + a * dt,0,2) # 速度变化
         # 限制在一定范围
        self.v = np.array([v * math.cos(self.orientation), v * math.sin(self.orientation)])
        self.pos = self.pos + np.array(self.v) * dt
        self.leader.set_position(self.pos[0],self.pos[1])
        self.leader.set_vel(self.v)
        self.leader.orientation=self.orientation
        return self.pos, self.v



class APFAgent_follower:
    def __init__(self, follower_agents, obstacles,leader_agent, goal_list, follower_id):
        self.k_rep = None
        self.k_att = None
        self.buffer = None
        self.list_of_agents = follower_agents
        self.leader_agent = leader_agent
        self.obstacles_list = obstacles
        self.id = follower_id
        self.pos = follower_agents[f"follower_{self.id}"].pos
        self.orientation = follower_agents[f"follower_{self.id}"].orientation
        self.v = follower_agents[f"follower_{self.id}"].vel
        self.goal = np.array(goal_list[self.id]) + self.leader_agent.pos # 绝对方位
        self.display_time = 0.1  # 时间步长

    # 与目标点引力
    def attractive_force(self):
        distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(self.goal))
        self.k_att = 1.0 + 0.1 * distance_to_goal  # 动态调整引力增益

        # 当接近目标点时，增加引力的强度
        if distance_to_goal < 20:
            self.k_att = max(20.0, self.k_att * 1.5)  # 增加引力增益

        attractive_force_ = self.k_att * distance_to_goal
        direction_vector = np.array(self.goal) - np.array(self.pos)
        if distance_to_goal > 0:
            normalized_direction = direction_vector / distance_to_goal
        else:
            normalized_direction = np.array([0, 0])

        attractive_force_x = attractive_force_ * normalized_direction[0]
        attractive_force_y = attractive_force_ * normalized_direction[1]

        return attractive_force_x, attractive_force_y

    # 与障碍斥力
    def repulsive_force(self):
        k_goal = 2
        distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(self.goal))
        self.buffer = 10 + 0.1 * distance_to_goal  # 动态调整斥力缓冲距离
        self.k_rep = 1.0 + 0.05 * distance_to_goal
        to_obs = []
        for i,value_ in enumerate(self.obstacles_list.values()):
            dis, theta = value_.calculate_dis_theta(self.pos[0],self.pos[1])
            r_obs = value_.radius
            dis = dis - r_obs  # 到障碍边界的距离
            to_obs.append([dis, theta])
        f_rep_x, f_rep_y = 0, 0
        for dis, theta in to_obs:
            # 斥力影响范围
            p0 = self.buffer
            if dis <= p0:
                # f_rep = k_rep * ((1 / dis - (1 / p0)) * (1 / dis) ** 2) # 原始斥力
                f_rep1 = self.k_rep * (1 / dis - 1 / p0) * distance_to_goal ** k_goal * (1 / dis) ** 2  # 改进斥力
                f_rep2 = k_goal / 2 * self.k_rep * (1 / dis - 1 / p0) ** 2 * distance_to_goal ** (k_goal - 1)
                direction_vector = np.array(self.goal) - np.array(self.pos)
                if distance_to_goal > 0:
                    normalized_direction = direction_vector / distance_to_goal
                else:
                    normalized_direction = np.array([0, 0])  # 如果距离为0，方向向量为0
                f_rep_x += f_rep1 * math.cos(theta) + f_rep2 * normalized_direction[0]
                f_rep_y += f_rep1 * math.sin(theta) + f_rep2 * normalized_direction[1]
        return f_rep_x, f_rep_y


    #多智能体内部的斥力
    def inter_agent_force(self):
        k_rep = 20 + 0.1 * np.linalg.norm(np.array(self.pos) - np.array(self.goal))
        k_goal = 2
        # 对领导者斥力
        dist = np.linalg.norm(np.array(self.leader_agent.pos-self.pos)) # 离领导者距离（障碍）
        theta = math.atan2(self.leader_agent.pos[1]-self.pos[1],self.leader_agent.pos[0]-self.pos[0]) # 离领导者角度(障碍)
        dist = dist - self.leader_agent.radius
        buffer = 5 #作用范围
        p0 = buffer
        distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(self.goal)) # 编队目标点
        direction_vector = np.array(self.goal) - np.array(self.pos)
        if distance_to_goal > 0:
            normalized_direction = direction_vector / distance_to_goal
        else:
            normalized_direction = np.array([0, 0])  # 如果距离为0，方向向量为0
        f_rep_x,f_rep_y = 0,0
        if dist <= p0:
            # f_rep = k_rep * ((1 / dis - (1 / p0)) * (1 / dis) ** 2) # 原始斥力
            f_rep1 = k_rep * (1 / dist - 1 / p0) * distance_to_goal ** k_goal * (1 / dist) ** 2  # 改进斥力
            f_rep2 = k_goal / 2 * k_rep * (1 / dist - 1 / p0) ** 2 * distance_to_goal ** (k_goal - 1)
            f_rep_x += f_rep1 * math.cos(theta) + f_rep2 * normalized_direction[0]
            f_rep_y += f_rep1 * math.sin(theta) + f_rep2 * normalized_direction[1]

        #与其他跟随者斥力
        for index, other_follower in enumerate(self.list_of_agents.values()):
            if index == self.id:
                continue
            else:
                dist_follower = np.linalg.norm(np.array(self.pos-other_follower.pos)) # 视其他智能体为障碍
                dist_follower = dist_follower - other_follower.radius
                theta_follower = math.atan2(self.pos[1]-other_follower.pos[1],self.pos[0]-other_follower.pos[0])
                if dist_follower <= p0:
                    f_rep1_follower = k_rep * (1 / dist_follower - 1 / p0) * distance_to_goal ** k_goal * (1 / dist_follower) ** 2
                    f_rep2_follower = k_goal / 2 * k_rep * (1 / dist_follower - 1 / p0) ** 2 * distance_to_goal ** (k_goal - 1)
                    f_rep_x += f_rep1_follower * math.cos(theta_follower) + f_rep2_follower * normalized_direction[0]
                    f_rep_y += f_rep1_follower * math.sin(theta_follower) + f_rep2_follower * normalized_direction[1]
        return f_rep_x,f_rep_y

    # Compute the total force acting on the agent at its current position
    # 合力
    def total_force(self):
        a_force_x, a_force_y = self.attractive_force()
        r_force_x, r_force_y = self.repulsive_force()
        i_force_x,i_force_y = self.inter_agent_force()
        # print("tot_fx:", force_x, "tot_fy:", force_y)
        force_x = a_force_x + r_force_x + i_force_x
        force_y = a_force_y + r_force_y + i_force_y

        return (force_x, force_y)

        # Move the agent towards the goal position based on the total force

    def move_towards(self):
        force_x, force_y = self.total_force()
        # # 单轨模型的参数
        # L = 2.5  # 前后轮间距，车辆的轴距
        # k_w = 3000  # 角速度参数
        # k_m = 100  # 加速度参数
        # dt = self.display_time  # 时间步长
        # force = np.linalg.norm(np.array([force_x, force_y]))  # 合力大小
        # print(force)
        # angle = np.arctan2(force_y, force_x) - self.orientation  # 合力方向与原方向之差
        # force_turn = force * math.sin(angle)  # 向心力
        # force_run = force * math.cos(angle)  # 牵引力
        # # print(force_run)
        # # print(force_turn)
        # a = np.clip(force_run / k_m, -0.2, 0.2)
        #
        #
        # # w_turn = np.clip(force_turn, -1, 1)
        # # a_run = np.clip(force_run, -1, 1)
        # w = np.clip(force_turn / k_w, -0.05, 0.05)
        # self.orientation = self.orientation + np.linalg.norm(self.v) / L * w * np.pi / 2 * dt
        # self.orientation = self.orientation % (2 * np.pi)
        # print(self.orientation)
        # last_v = np.linalg.norm(self.v)
        # v = np.clip(last_v + a * dt, 0, 1)   # 速度变化
        # # 限制在一定范围
        # self.v = np.array([v * math.cos(self.orientation), v * math.sin(self.orientation)])
        # self.pos = self.pos + np.array(self.v) * dt
        # self.list_of_agents[f"follower_{self.id}"].set_position(self.pos[0],self.pos[1])
        # self.list_of_agents[f"follower_{self.id}"].set_vel(self.v)
        # self.list_of_agents[f"follower_{self.id}"].orientation = self.orientation

        return self.pos,self.v


class APF_circle_agent():
    def __init__(self, radius=5, pos=np.array([25, 25]), vel=np.array([0, 0]), orientation=np.pi/4,):
        self.radius = radius
        self.pos = pos
        self.vel = vel
        self.orientation = orientation
        self.target = False

    def set_position(self, x, y):
        self.pos[0] = x
        self.pos[1] = y

    def set_vel(self, v):
        self.vel = v

    def position(self):
        return self.pos




class APFEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, width=100, height=100, num_obstacles=15, agent_radius=1, obs_radius=2, safe_theta=8,
                 target_radius=4,
                 target_pos=np.array([50, 50]), delta=0.1, follower_uav_num=3):
        super().__init__()

        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obs_radius = obs_radius
        self.target_radius = target_radius
        self.agent_radius = agent_radius
        self.obs_delta = 10
        self.fig = None
        self.ax = None

        self.display_time = delta
        self.safe_theta = safe_theta
        self.step_time = 0
        self.follower_uav_num = follower_uav_num
        self.last_leader_action = [0, 0]
        self.leader_agent_trajectory = []
        self.follower_agent_trajectory = []
        for _ in range(self.follower_uav_num):
            self.follower_agent_trajectory.append([])
        self.last_follower_action = [[0, 0], [0, 0], [0, 0]]
        # 生成circle_agent实例列表
        self.leader_uav = APF_circle_agent(pos=np.array([25, 25]), vel=np.array([0, 0]), orientation=np.pi/4)

        self.follower_uavs = {}

        self.formation_pos = [
            [0, 2 * 3],
            [-1.732 * 3, -1 * 3],
            [1.732 * 3, -1 * 3]
        ]

        for i in range(follower_uav_num):
            self.follower_uavs[f"follower_{i}"] = APF_circle_agent(radius=self.agent_radius,
                                                               pos=np.array([self.leader_uav.pos[0] + self.formation_pos[i][0],
                                                                    self.leader_uav.pos[1] + self.formation_pos[i][1]]),
                                                               vel=np.array([0, 0]),orientation=np.pi/4)


        self.fix_position = [
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
        for i in range(len(self.fix_position)):#固定障碍
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

        self.leader_target_pos = target_pos  # 目标位置
        self._check_obs()
        self.reset()


        self.APFAgent_leader = APFAgent_leader(self.follower_uavs, self.obstacles,self.leader_uav,self.leader_target_pos)
        self.APFAgent_follower_0 = APFAgent_follower(self.follower_uavs,self.obstacles,
                                                     self.leader_uav,self.formation_pos,follower_id=0)
        self.APFAgent_follower_1 = APFAgent_follower(self.follower_uavs, self.obstacles,
                                                     self.leader_uav, self.formation_pos, follower_id=1)
        self.APFAgent_follower_2 = APFAgent_follower(self.follower_uavs, self.obstacles,
                                                     self.leader_uav, self.formation_pos, follower_id=2)

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
            if dis <= self.obs_radius + self.agent_radius + self.safe_theta / 2:
                return True
        for i in range(self.follower_uav_num):
            for obs in self.obstacles.values():
                dis = np.linalg.norm(obs.position() - self.follower_uavs[f"follower_{i}"].position())
                if dis <= self.obs_radius + self.agent_radius + self.safe_theta / 2:
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
            if dis < self.obs_radius + self.target_radius + self.safe_theta / 2:
                return True
        return False

    def _check_fix_obs_agent(self, leader_pos):
        obstacles_keys = list(self.obstacles.keys())
        obstacles_list = list(self.obstacles.values())

        # 假设前9个障碍物是固定的
        fixed_obstacles = obstacles_list[:9]
        random_obstacles = obstacles_list[9:]

        for obs in fixed_obstacles:
            if np.linalg.norm(
                    obs.position() - np.array(leader_pos)) < self.agent_radius + self.obs_radius + self.safe_theta / 4:
                return True

            for i in range(self.follower_uav_num):
                if np.linalg.norm(obs.position() - np.array(self.follower_uavs[
                                                                f"follower_{i}"].position())) < self.agent_radius + self.obs_radius + self.safe_theta / 4:
                    return True

            return False

    def reset(self):

        # 随机位置的障碍物
        for i in range(len(self.fix_position), self.num_obstacles):
            self.obstacles[f"obstacle_{i}"].pos_x = np.random.rand() * self.width * 0.7 + self.width * 0.15
            self.obstacles[f"obstacle_{i}"].pos_y = np.random.rand() * self.height * 0.7 + self.height * 0.15

        self.leader_target_pos = [self.width * 0.1 + np.random.rand() * self.width * 0.8,
                                  self.height * 0.1 + np.random.rand() * self.height * 0.8]
        # self.leader_target_pos = [85 + np.random.rand() * 5 , 85 + np.random.rand() * 5]
        self.leader_uav.set_position(self.width * 0.10 + np.random.rand() * self.width * 0.8,
                                       self.height * 0.10 + np.random.rand() * self.height * 0.8)

        # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)

        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_position(self.leader_uav.pos[0] + self.formation_pos[i][0],
                                                             self.leader_uav.pos[1] + self.formation_pos[i][1])

        # 确保开始不与固定障碍物碰撞
        flag0 = self._check_fix_obs_agent(self.leader_uav.pos)
        while flag0:
            self.leader_uav.set_position(self.width * 0.10 + np.random.rand() * self.width * 0.8,
                                           self.height * 0.10 + np.random.rand() * self.height * 0.8)
            # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)

            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_uav.pos[0] + self.formation_pos[i][0],
                                                                 self.leader_uav.pos[1] + self.formation_pos[i][1])
            flag0 = self._check_fix_obs_agent(self.leader_uav.pos)

        # 确保目标不与固定障碍物碰撞
        flag2 = self._check_obs_target(self.leader_target_pos)
        while flag2:
            self.leader_target_pos = [self.width * 0.1 + np.random.rand() * self.width * 0.8,
                                      self.height * 0.1 + np.random.rand() * self.height * 0.8]
            # self.leader_target_pos = [85 + np.random.rand() * 5 , 85 + np.random.rand() * 5]

            flag2 = self._check_obs_target(self.leader_target_pos)

        self._check_obs()

        flag1 = self._check_obs_agent(self.leader_uav)
        while np.linalg.norm(np.array(self.leader_target_pos) - np.array(
                self.leader_uav.pos)) < self.agent_radius + self.target_radius + self.safe_theta * 3 or flag1:
            self.leader_uav.set_position(self.width * 0.10 + np.random.rand() * self.width * 0.8,
                                           self.height * 0.10 + np.random.rand() * self.height * 0.8)
            # self.leader_agent.set_position(np.random.rand() * 10 + 5, np.random.rand() * 10 + 5)

            for i in range(self.follower_uav_num):
                self.follower_uavs[f"follower_{i}"].set_position(self.leader_uav.pos[0] + self.formation_pos[i][0],
                                                                 self.leader_uav.pos[1] + self.formation_pos[i][1])
            flag1 = self._check_obs_agent(self.leader_uav)

        goal_flag=self.goal_flag()
        self.leader_uav.target = self.leader_target_pos
        if all(goal_flag):
            done = True
        else:
            done = False
        self.leader_uav.set_vel(0)
        delta_x = self.leader_target_pos[0] - self.leader_uav.pos[0]
        delta_y = self.leader_target_pos[1] - self.leader_uav.pos[1]
        # 处理零向量
        if delta_x == 0 and delta_y == 0:
            theta = 0
        else:
            theta = math.atan2(delta_y, delta_x)
        # 调整到 0 到 2π
        if theta < 0:
            theta += 2 * math.pi
        self.leader_uav.orientation = theta
        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].orientation=theta
        return self.get_all_pos(),self.get_all_vel(),done

    def get_all_pos(self):
        return [self.leader_uav.pos,list(self.follower_uavs.values())[0].pos,list(self.follower_uavs.values())[1].pos,
                 list(self.follower_uavs.values())[2].pos]

    def get_all_vel(self):
        return [self.leader_uav.vel,list(self.follower_uavs.values())[0].vel,
                 list(self.follower_uavs.values())[1].vel,list(self.follower_uavs.values())[2].vel]

    def goal_flag(self): #达到目标标志
        target_distance = np.linalg.norm(np.array(self.leader_target_pos) - np.array(self.leader_uav.pos))

        if target_distance <=4:
            self.leader_uav.target = True
        else:
            self.leader_uav.target = False
        for i in range(self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].set_vel([0, 0])
            dis_to_goal = np.linalg.norm(
                np.array(self.leader_uav.pos) - np.array(self.follower_uavs[f"follower_{i}"].pos) - np.array(
                    self.formation_pos))
            if self.leader_uav.target and dis_to_goal <= 0.05:
                self.follower_uavs[f"follower_{i}"].target = True
            else:
                self.follower_uavs[f"follower_{i}"].target = False
        goal_flag = [self.leader_uav.target, self.follower_uavs[f"follower_{0}"].target,
                     self.follower_uavs[f"follower_{1}"].target, self.follower_uavs[f"follower_{2}"].target]
        return goal_flag

    def step(self):
        """
            所有智能体前进一步
        """
        goal_flag=self.goal_flag()
        if not goal_flag[0]:
            leader_pos,leader_v=self.APFAgent_leader.move_towards()
        if not goal_flag[1]:
            follower0_pos,follower0_v=self.APFAgent_follower_0.move_towards()
        if not goal_flag[2]:
            follower1_pos, follower1_v = self.APFAgent_follower_1.move_towards()
        if not goal_flag[3]:
            follower2_pos, follower2_v = self.APFAgent_follower_2.move_towards()

        if all(goal_flag):
            done=True
        else:
            done=False

        return self.get_all_pos(),self.get_all_vel(),done

    def render(self, display_time=0.1):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        leader_agent = patches.Circle(tuple(self.leader_uav.pos), self.agent_radius, color='purple', fill=True)
        self.ax.add_patch(leader_agent)
        arrow_length = self.agent_radius * 1  # Adjust length as needed
        arrow_dx = arrow_length * np.cos(self.leader_uav.orientation)
        arrow_dy = arrow_length * np.sin(self.leader_uav.orientation)
        arrow = patches.FancyArrow(
            float(self.leader_uav.pos[0]),
            float(self.leader_uav.pos[1]),
            arrow_dx,
            arrow_dy,
            width=self.agent_radius * 0.25,
            color='purple'
        )
        self.ax.add_patch(arrow)
        # 记录智能体当前的位置到轨迹
        self.leader_agent_trajectory.append(self.leader_uav.pos.copy())
        # 绘制智能体的轨迹
        if len(self.leader_agent_trajectory) > 1:
            traj_x, traj_y = zip(*self.leader_agent_trajectory)
            self.ax.plot(traj_x, traj_y, color='blue', linestyle='-', marker='o', markersize=1, label='Trajectory')

        colors = ['orange', 'cyan', 'magenta']  # Colors for each follower

        # 绘制跟随者无人机
        for i in range(self.follower_uav_num):
            uav = patches.Circle((self.follower_uavs[f"follower_{i}"].pos[0],self.follower_uavs[f"follower_{i}"].pos[1]), self.agent_radius, color='orange', fill=True)
            self.ax.add_patch(uav)
            self.follower_agent_trajectory[i].append(self.follower_uavs[f"follower_{i}"].pos.copy())

            # Draw trajectory for this follower
            if len(self.follower_agent_trajectory[i]) > 1:
                traj_x, traj_y = zip(*self.follower_agent_trajectory[i])
                self.ax.plot(traj_x, traj_y, color=colors[i % len(colors)], linestyle='-', marker='o', markersize=1,
                             label=f"Follower {i} Trajectory")

        # 绘制障碍物
        obses = [patches.Circle((obs.pos_x, obs.pos_y), self.obs_radius, color='red', fill=True) for obs in
                 self.obstacles.values()]
        for obs_circle in obses:
            self.ax.add_patch(obs_circle)
        # 绘制目标
        target = patches.Circle(tuple(self.leader_target_pos), self.target_radius, color='green', fill=True)
        self.ax.add_patch(target)

        plt.pause(display_time)  # 暂停以更新图形
        # plt.show()

    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None


# Example usage
env = APFEnv(width=100, height=100, num_obstacles=15, agent_radius=1, obs_radius=2, safe_theta=8, target_radius=4, target_pos=np.array([80, 80]), delta=0.1, follower_uav_num=3)
env.render(display_time=0.01)
for _ in range(1000):
    env.step()
    env.render(display_time=0.01)
env.render_close()
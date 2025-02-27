# -*- coding: utf-8 -*-
from .rvo_inter import rvo_inter
import numpy as np
from .obstacle import obstacle
from .circle_agent_sac import circle_agent
from .follower_uav import follower_uav
# import matplotlib.pyplot as plt
from math import sin, cos, tan, pi, sqrt, log
import matplotlib.patches as patches
import torch
from env_formation.sac import SAC
import matplotlib
# matplotlib.use('Agg')  # 必须放在 import matplotlib.pyplot 之前
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
# plt.ion() 
# plt.rcParams['toolbar'] = 'None' 


np.set_printoptions(precision=5, suppress=True)
class Custom:
    metadata = {'render.modes':['human']}

    def __init__(self, width=100, height=100, num_obstacles=15, agent_radius=1, obs_radius = 2,safe_theta = 8,target_radius = 4,
                 target_pos = np.array([50, 50]), delta = 0.1, memo_size=100000, follower_uav_num = 1):
        super().__init__()
        self.width = width
        self.height = height
        # self.num_obstacles = num_obstacles
        # self.obs_radius = obs_radius
        self.target_pos = target_pos
        self.target_radius = target_radius
        self.agent_radius = agent_radius
        # self.obs_delta =10
        self.fig = None
        self.ax = None
        
        self.display_time = delta
        self.safe_theta = safe_theta
        self.rvo_inter = rvo_inter()
        self.follower_uav_num = follower_uav_num
        self.follower_uav_num_max = 3
        self.last_leader_action = [0, 0]
        self.leader_agent_trajectory = []
        self.follower_agent_trajectory = []
        for _ in range (self.follower_uav_num):
            self.follower_agent_trajectory.append([])
        self.last_follower_action = [[0, 0], [0, 0], [0, 0]]

        # # 生成circle_agent实例列表
        # self.leader_agent = circle_agent(self, pos=[25, 25], vel=[0,0], orientation=0, memo_size=memo_size,
        #                                 #  state_dim=13 + self.num_obstacles * 5,
        #                                  state_dim=13 + 6 * 5,
        #                                  action_dim=2,
        #                                  alpha=1e-4,
        #                                  beta=1e-4,
        #                                  hidden_dim=512,
        #                                  gamma=0.99,
        #                                  tau=0.01,
        #                                  batch_size=512,
        #                                  target_entropy= -log(2))
        
        self.follower_uavs = {}

        self.formation_pos = [
            [0, 2*3],
            [-1.732*3, -1*3],
            [1.732*3, -1*3]
        ]

        for i in range (follower_uav_num):
            self.follower_uavs[f"follower_{i}"] = follower_uav(radius=self.agent_radius,
                                                               pos = [10+np.random.rand()*80, 10+np.random.rand()*80],
                                                               vel=[0,0],
                                                               memo_size=100000, state_dim=40, action_dim=2, alpha=1e-4, beta=1e-4,
                                                               alpha_lr=1e-4, hidden_dim=512, gamma=0.99, tau=0.01, batch_size=512,target_entropy=-log(2) )
        
        self.SAC = SAC(state_dim = (4+2+3*5+2+5*(self.follower_uav_num_max-1) + 2 ),
                        #    state_dim = (4+4+5*self.num_obstacles+2+5*(self.follower_uav_num-1))
                                                            hidden_dim = 512,
                                                            action_dim=2,
                                                            actor_lr=1e-4,
                                                            critic_lr=5e-5,
                                                            alpha_lr=1e-5,
                                                            # alpha_lr=torch.tensor(np.log(0.1), dtype=torch.float, requires_grad=True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                                                            # target_entropy= -log(2*self.follower_uav_num),
                                                            target_entropy= -log(2),
                                                            tau=0.005,
                                                            gamma=0.99,
                                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                                            agent_num=self.follower_uav_num)
        self.reset()
    
    def reset(self):
        self.target_pos = np.random.rand(2)*self.width*0.6 + 20
        

        for i in range(self.follower_uav_num):
            new_position = np.random.rand(2)*self.width*0.8 + 10
            pos_flag = False
            while pos_flag == False:
                if np.linalg.norm(new_position - self.target_pos) < self.width*0.4:
                    new_position = np.random.rand(2)*self.width*0.8 + 10
                else: 
                    pos_flag = True

            self.follower_uavs[f"follower_{i}"].set_position(new_position[0], new_position[1])
            self.follower_uavs[f"follower_{i}"].done = False
            self.follower_uavs[f"follower_{i}"].target = False
            self.follower_uavs[f"follower_{i}"].side_done = False
            self.follower_uavs[f"follower_{i}"].uav_done = False
            self.follower_uavs[f"follower_{i}"].obs_done = False
            self.follower_uavs[f"follower_{i}"].formation_done = False
            self.follower_uavs[f"follower_{i}"].set_vel(0, 0)
            self.last_follower_action[i] = [0, 0]
        
        self.observe()
    
    def observe(self):
        """
        update follower's observation
        """
        for i in range(self.follower_uav_num):
            side_pos_2 = [
                round(self.follower_uavs[f"follower_{i}"].pos[0] / self.width, 5),
                round((self.width - self.follower_uavs[f"follower_{i}"].pos[0]) / self.width, 5),
                round(self.follower_uavs[f"follower_{i}"].pos[1] / self.height, 5),
                round((self.height - self.follower_uavs[f"follower_{i}"].pos[1]) / self.height, 5),
            ]
            target_pos_2 = [
                round(self.target_pos[0] - self.follower_uavs[f"follower_{i}"].pos[0], 5),
                round(self.target_pos[1] - self.follower_uavs[f"follower_{i}"].pos[1], 5)
            ]

            obs_pos_vel_2 = []
            for j in range (3):
                obs_pos_vel_2.extend([-1, -1, -1, -1, False])
            
            leader_vel = [0, 0] #TODO
            
            follower_pos_ = []
            for j in range (2):
                follower_pos_.extend([0, 0, 0, 0, False]) #TODO
            
            self_pos_2 = [
                round(self.follower_uavs[f"follower_{i}"].vel[0], 5),
                round(self.follower_uavs[f"follower_{i}"].vel[1], 5)
            ]
            
            self.follower_uavs[f"follower_{i}"].observation = np.array(
                side_pos_2 +
                target_pos_2 +
                obs_pos_vel_2 +
                self_pos_2 +
                follower_pos_ +
                leader_vel 
            )
    
    def step(self, follower_actions, last_follower_goal_distance):
        """
        follower step
        """
        self._apply_follower_action(follower_actions)

        self.observe()
        follower_observations = []
        follower_done = []
    
        for i in range (self.follower_uav_num):
            next_state = self.follower_uavs[f"follower_{i}"].observation
            follower_observations.append(next_state)

            if self.follower_uavs[f"follower_{i}"].done == True:
                follower_done.append(True)
            else:
                follower_done.append(False)

        follower_reward = self._caculate_follower_reward(follower_actions, last_follower_goal_distance)

        return follower_observations, follower_reward, follower_done
    
    def _apply_follower_action(self, actions):
        for i in range (self.follower_uav_num):
            self.follower_uavs[f"follower_{i}"].vel[0] = actions[2*i] * 2.5
            self.follower_uavs[f"follower_{i}"].vel[1] = actions[2*i+1] * 2.5

            dx = self.follower_uavs[f"follower_{i}"].vel[0] * self.display_time
            dy = self.follower_uavs[f"follower_{i}"].vel[1] * self.display_time

            self.follower_uavs[f"follower_{i}"].pos[0] += dx
            self.follower_uavs[f"follower_{i}"].pos[1] += dy

            target_dis, _ = Custom.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{i}"].pos, self.target_pos)
            # print(self.follower_uavs[f"follower_{i}"].pos)
            # print()
            # print(target_dis)
            # if target_dis < self.agent_radius + self.target_radius:
            if target_dis < 1:
                self.follower_uavs[f"follower_{i}"].target = True
                self.follower_uavs[f"follower_{i}"].done = True
            if self.follower_uavs[f"follower_{i}"].pos[0]<self.agent_radius or \
            self.follower_uavs[f"follower_{i}"].pos[0]>self.width - self.agent_radius or \
            self.follower_uavs[f"follower_{i}"].pos[1]<self.agent_radius or \
            self.follower_uavs[f"follower_{i}"].pos[1]>self.height - self.agent_radius:
                self.follower_uavs[f"follower_{i}"].done = True
                self.follower_uavs[f"follower_{i}"].side_done = True
    

    def _caculate_follower_reward(self, follower_actions, last_follower_goal_distance):
        follower_rewards = []
        for i in range (self.follower_uav_num):
            follower_action = follower_actions[2*i: 2*i+2]
            last_follower_goal_dist = last_follower_goal_distance[i]
            side_reward = self._follower_side_reward(i)
            target_reward = self._follower_target_reward(i, last_follower_goal_dist)
            vel_reward = self._follower_vel_reward(i, follower_action)
            # print(f"side_reward: {side_reward:.2f}, target_reward: {target_reward:.2f}, vel_reward: {vel_reward:.2f}")
            r = round(side_reward + target_reward + vel_reward, 5)
            follower_rewards.append(r)

            self.last_follower_action[i] = follower_action
        return follower_rewards
    
    def _follower_target_reward(self, uav_id, last_follower_goal_dis):
        if self.follower_uavs[f"follower_{uav_id}"].target == True:
            # self.follower_uavs[f"follower_{uav_id}"].target = False
            # self.follower_uavs[f"follower_{uav_id}"].done = False
            return 100
        # if  self.follower_uavs[f"follower_{uav_id}"].formation_done:
                # return round(-40, 5)
        else :
            # return 0
            dis, angle = Custom.calculate_relative_distance_and_angle(self.follower_uavs[f"follower_{uav_id}"].pos,
                                                                        self.target_pos)
            # print(f"follower_{uav_id} goal dis :", dis)
            # return -(dis - last_follower_goal_dis) *100 - dis * 50
            # if dis <= 5.0:
            #     # return round(250 /(dis + 1), 5)
            #     # return round(5, 5)
            #     round(10 / (dis), 5)  # 奖励与距离反比
      
            # if dis >= self.obs_delta*3:
            # if dis >= 80:
                # return round(-20, 5)
            r=0
            if dis >= 10:
                r = round(- dis *dis/20, 5)
            else:
                r = round(50 / (dis) - 10, 5)  # 奖��与距离反比
            return  r
            # return  round(200/dis, 5)


    def _follower_side_reward(self, uav_id):
        reward = 0
        distances = [self.follower_uavs[f"follower_{uav_id}"].pos[0], \
                                self.width - self.follower_uavs[f"follower_{uav_id}"].pos[0], \
                                self.follower_uavs[f"follower_{uav_id}"].pos[1], \
                                self.height - self.follower_uavs[f"follower_{uav_id}"].pos[1]]
        for dis in distances:
            if dis > self.width * 0.1:
                re = 0
            
            elif self.width *0.05 < dis <= self.width* 0.1:
                # re = round(-self.width *0.05 /dis, 5)
                re = round(-10 / (dis), 5)
            elif dis <= self.width *0.05:
                # if self.leader_agent.done and not self.leader_agent.target:
                #     return -200
                # re = round(-20 *((self.width *0.05)/(self.width *0.05 + dis)), 5)
                re = max(round(-60 / (dis), 5), -60)

            reward += re
        # return round(reward *40, 5)
        return round(reward, 5)
    
    def _follower_vel_reward(self, uav_id, action):
        reward = 0
        reward += -abs(action[0]) - abs(action[1])

        return round(reward, 5)


    def render(self, display_time = 0.1):
        # plt.ion() 
        # print(self.width, self.height)
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10,10), dpi=100)
            # print("1111")

            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        
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

        
        
        # 绘制目标
        target = patches.Circle(self.target_pos, self.target_radius, color='green', fill=True)
        self.ax.add_patch(target)

        plt.pause(display_time)  # 暂停以更新图形
        # plt.show()
    def render_close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
        for i in range(self.follower_uav_num):
            self.follower_agent_trajectory[i] = []


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


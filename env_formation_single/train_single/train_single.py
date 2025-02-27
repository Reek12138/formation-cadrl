import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys
from numpy import inf
import torch.nn.functional as F
from numpy import sqrt
from math import sin, cos, tan, pi, sqrt, log
np.set_printoptions(precision=5, suppress=True)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from env_formation.env_follow import Custom
from env_formation.circle_agent_sac import circle_agent, ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

NUM_EPISODE = 90002
NUM_STEP = 3000
MEMORY_SIZE = 100000
# BATCH_SIZE = 512
BATCH_SIZE = 1024
TARGET_UPDATE_INTERVAL= 2
RENDER_FREQUENCE = 500
RENDER_NUM_EPISODE = 200
RENDER_NUM_STEP = 1000
BREAK_FLAG = False

env = Custom(delta=0.1)
env.reset()

scenario = "single_follower_train"
current_path = os.path.dirname(os.path.realpath(__file__))
# agent_path = current_path + "/leader_model/"
# better_path = current_path + "/leader_model/better/"
follower_path = current_path + "/follower_model"
follower_better_path = current_path + "/follower_model/better/"

env.SAC.replay_buffer.clear()

highest_eposide_reward = -inf
eval_highest_num_reach_goal = 0
episode_reward = 0
BREAK_FLAG= False

for episode_i in range (NUM_EPISODE):
    if BREAK_FLAG == True:
        break

    if episode_i > 0:
        print(f"episode {episode_i+1}   ===========    LAST EPISODE REWARD : {episode_reward:.2f} STEP : {num_step} ")

    env.reset()

    # initialize
    last_target_dis = []
    last_follower_vel = []
    for i in range (env.follower_uav_num):
        last_target_dis.append(np.linalg.norm(env.target_pos - env.follower_uavs[f"follower_{i}"].pos))
        last_follower_vel.append(env.follower_uavs[f"follower_{i}"].vel)

    


    episode_reward = 0
    num_step = 0
    for step_i in range(NUM_STEP):
        num_step += 1
        total_step = episode_i * NUM_STEP + step_i
        follower_actions = []
        follower_observations = []
        for i in range (env.follower_uav_num):
            action = env.SAC.take_action(env.follower_uavs[f"follower_{i}"].observation)
            noisy_action = action + np.random.normal(0, 0.2, size=action.shape)
            noisy_action = np.clip(noisy_action, -1, 1)
            follower_actions.extend(noisy_action)
            follower_observations.append(env.follower_uavs[f"follower_{i}"].observation)

        next_observations, rewards, dones = env.step(follower_actions, last_target_dis)
        episode_reward += sum(rewards)/env.follower_uav_num

        for i in range (env.follower_uav_num):
            follower_S = follower_observations[i]
            follower_A = np.array(follower_actions[2*i : 2*i+2])
            follower_R = np.array(rewards[i])
            follower_NS = next_observations[i]
            follower_done_ = dones[i]
            env.SAC.replay_buffer.add(state=follower_S, action=follower_A, reward=follower_R, next_state=follower_NS, done=follower_done_)
        
        current_memo_size = min(MEMORY_SIZE, total_step)
        batch_flag = current_memo_size >= BATCH_SIZE * 50
        # if(total_step +1)% TARGET_UPDATE_INTERVAL == 0 and batch_flag == True:
        if(total_step +1)% TARGET_UPDATE_INTERVAL == 0 and env.SAC.replay_buffer.size() >= BATCH_SIZE*50:
            # print("sample")
            fs, fa, fr, fns, fd = env.SAC.replay_buffer.sample(batch_size=BATCH_SIZE)
            f_transition_dict = {'states' : fs,
                                'actions' : fa,
                                'rewards' : fr,
                                'next_states' : fns,
                                'dones' : fd}
            env.SAC.update(transition_dict= f_transition_dict)
        
        # collision_detection
        if any(dones):
            index = dones.index(True)
            if env.follower_uavs[f"follower_{index}"].target == True:
                print(f"\033[92m******************** REACH GOAL ********************step{step_i} \033[0m")
            if env.follower_uavs[f"follower_{index}"].side_done == True:
                print(f"\033[91mxxxxxxxxxxxxxxxxxx  FOLLOWER SIDE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
            break         

#====== 每个episode结束后更新一次=====================================================================
    if env.SAC.replay_buffer.size() >= BATCH_SIZE*50:
        # print("sample")
        fs, fa, fr, fns, fd = env.SAC.replay_buffer.sample(batch_size=BATCH_SIZE)
        f_transition_dict = {'states' : fs,
                            'actions' : fa,
                            'rewards' : fr,
                            'next_states' : fns,
                            'dones' : fd}
        env.SAC.update(transition_dict= f_transition_dict)

        if not os.path.exists(follower_path):
            os.makedirs(follower_path)
        env.SAC.save_model(follower_path, scenario)

# evaluation
    if episode_i > 0 and episode_i % RENDER_FREQUENCE == 0:
        eval_env = Custom(delta=0.1)
        num_reach_goal = 0
        num_side_collision = 0
        print("======================验证中=========================")
        for test_episode_i in range(RENDER_NUM_EPISODE):
            eval_env.SAC.load_model(follower_path, scenario)
            eval_env.reset()

            for test_step_i in range(RENDER_NUM_STEP):
                for i in range (eval_env.follower_uav_num):
                    action = eval_env.SAC.take_action(eval_env.follower_uavs[f"follower_{i}"].observation)
                    noisy_action = action + np.random.normal(0, 0.2, size=action.shape)
                    noisy_action = np.clip(noisy_action, -1, 1)
                    follower_actions.extend(noisy_action)
                    follower_observations.append(eval_env.follower_uavs[f"follower_{i}"].observation)

                next_observations, rewards, dones = eval_env.step(follower_actions, last_target_dis)

                if any(dones):
                    index = dones.index(True)
                    if eval_env.follower_uavs[f"follower_{index}"].target == True:
                        num_reach_goal += 1
                        print(f"\033[92m******************** REACH GOAL ********************step{step_i} \033[0m")
                    if eval_env.follower_uavs[f"follower_{index}"].side_done == True:
                        num_side_collision += 1
                        print(f"\033[91mxxxxxxxxxxxxxxxxxx  FOLLOWER SIDE COLLISION xxxxxxxxxxxxxxxxxxx step{step_i} \033[0m")
                    break
        if num_reach_goal >= eval_highest_num_reach_goal:
            if not os.path.exists(follower_better_path):
                os.makedirs(follower_better_path)
            eval_env.SAC.save_model(follower_better_path, scenario)

            print(f"-----reach goal num = {num_reach_goal}")
            print("--------------------------跟随者更好的参数-----------------")
        
        if num_reach_goal == RENDER_NUM_EPISODE:
            BREAK_FLAG = True
        


import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.cuda
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from env_formation.env_formation_single import CustomEnv
from env_formation.env_follow import Custom
from env_formation.circle_agent_sac import circle_agent, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

RENDER_EPISODE_NUM = 5
RENDER_NUM_STEP = 1000

scenario = "single_follower_train"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/leader_model/"
better_path = current_path + "/leader_model/better/"
follower_path = current_path + "/follower_model/"
follower_better_path = current_path + "/follower_model/better/"
env = Custom(delta=0.1)

for episode_i in range(RENDER_EPISODE_NUM):
    # env.leader_agent.sac_network.load_model(better_path, scenario)
    # env.leader_agent.sac_network.load_model(agent_path, scenario)
    # env.SAC.load_model(follower_better_path, scenario)
    # env.SAC.load_model(follower_path, scenario)
    env.SAC.load_model(follower_better_path, scenario)

    env.reset()

    # target_distance = np.linalg.norm(np.array(env.leader_agent.pos) - np.array(env.leader_target_pos))
    print("rendering episode ", episode_i, "=============")
    step_break_flag = False
    for step_i in range (RENDER_NUM_STEP):
        # time.sleep(0.5)
        # print("==============================")
        last_target_dis = []
        last_follower_vel = []
        for i in range (env.follower_uav_num):
            last_target_dis.append(np.linalg.norm(env.target_pos - env.follower_uavs[f"follower_{i}"].pos))
            last_follower_vel.append(env.follower_uavs[f"follower_{i}"].vel)
            if env.follower_uavs[f"follower_{i}"].done == True:
                # print("done")
                step_break_flag = True
                break
        if step_break_flag == True:
            break
              
        follower_actions = []
        follower_observations = []
        for i in range (env.follower_uav_num):
            action = env.SAC.take_action(env.follower_uavs[f"follower_{i}"].observation)
            noisy_action = action + np.random.normal(0, 0.2, size=action.shape)
            noisy_action = np.clip(noisy_action, -1, 1)
            follower_actions.extend(noisy_action)
            follower_observations.append(env.follower_uavs[f"follower_{i}"].observation)

        next_observations, rewards, dones = env.step(follower_actions, last_target_dis)
        # print("111111")
        env.render(display_time = 0.01)
    env.render_close()
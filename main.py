# main.py
from Franka_env import FrankaRLEnv  # 導入剛才那個環境
from ddpg_torch import Agent  # 導入你的 DDPG 算法
import numpy as np


env = FrankaRLEnv(FLAGS)


agent = Agent(alpha=0.0001, beta=0.001, input_dims=[7],
              tau=0.005, batch_size=256, n_actions=3)

# 開始訓練循環
for i in range(1000):
    obs_dict = env.reset()
    observation = obs_dict['robot_state']
    done = False

    while not done:

        action = agent.choose_action(observation)


        next_obs_dict, reward, done, info = env.step(action)
        next_observation = next_obs_dict['robot_state']

 
        agent.remember(observation, action, reward, next_observation, done)
        agent.learn()

        observation = next_observation
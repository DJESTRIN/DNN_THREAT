# import gym
# env=gym.make("ALE/SpaceInvaders-v5")
# o=env.reset()
# for _ in range(1000):
#     env.step(env.action_space.sample())
#     env.render('human')
# env.close()  # https://github.com/openai/gym/issues/893

# # import atari_py
# # print(atari_py.list_games())


import gym
from gym import envs
import matplotlib.pyplot as plot
import numpy as np
import ipdb
import time

for key in list(envs.registry.keys()):
    # Create the environment
    env = gym.make(key, render_mode="human")
    print(env)
    # Run the environment for a fixed number of episodes
    num_episodes = 1
    for i_episode in range(num_episodes):
        observation, dummy = env.reset()
        # ipdb.set_trace()
        observation = np.array(observation)
        print(observation.shape)
        plot.figure()
        plot.imshow(np.array(observation))
        x = key + ".png"
        x = x.replace("/","_")
        plot.savefig(x)
        for t in range(100):
            env.render()  # Render the environment in a GUI
            # time.sleep(0.1)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)[:4]
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                env.close()  # Close the environment after each episode
                break
        env.close()  # Close the environment after each episode
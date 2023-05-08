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
import os
import random
from PIL import Image
from classify_observation import Classify_observation

folder = "TestingFolder"
numberOfImages = 100

def getFileCount():
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


# Create the environment
key = "BattleZone-v4"
cob = Classify_observation("C:\Program Files\ilastik-1.4.0", "BattleZoneVersion0.ilp", "BattleZone")
env = gym.make(key, render_mode="human")
print(env)
# Run the environment for a fixed number of episodes
num_episodes = 3
episode = 1
# while getFileCount() <= numberOfImages:
while episode <= num_episodes:
    observation, dummy = env.reset()
    # ipdb.set_trace()
    observation = np.array(observation)
    # plot.axis('off')
    t = 0
    randomness = 0.05
    file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

    # while getFileCount() <= numberOfImages:
    while True:
        env.render()  # Render the environment in a GUI
        # time.sleep(0.1)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)[:4]
        x = key + "Image" + str(t)
        x = x.replace("/","_")
        x = folder + "/" + x
        random_num = random.random() # (0,1]
        if random_num <= randomness:
            # fig = plot.figure(frameon=False)
            # plot.imshow(np.array(observation))
            # plot.savefig(x)
            Image.fromarray(observation).save('{}.png'.format(x))
            plot.close()
            print("Images in {}: {}".format(folder,getFileCount()))

        t += 1
        file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.close()  # Close the environment after each episode
            
            with open("testtext.txt", "a") as textfile:
                textfile.write("\n\nEpisode {}:\n".format(episode))
            episode += 1

            cob.exportClassesFromDir("Exports", folder)
            cob.displayAllClassesInOutputTextFile("Exports", "testtext.txt", "Enemy")
            for file in [file for file in os.listdir(folder) if file.endswith('png')]:
                os.remove(os.path.join(folder, file))
            break
    env.close()  # Close the environment after each episode


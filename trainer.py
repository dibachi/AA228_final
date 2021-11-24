#trainer.py

import gym
#set environment
env = gym.make('BipedalWalker-v3')
num_eps = 20 #number of episodes
num_steps = 100 #number of time steps per episode
for i_episode in range(num_eps):
    #in essence, the episodes are exploration steps. Opportunities
    # to train a new, improved policy based on the successes and failures
    # of the previous episode. To that end, each time step is one exploitation step
    observation = env.reset()
    ###################################################
    #start with random policy if @ episode 0 or use 
    #trained policy using data from previous episode(s)
    ###################################################
    for t in range(num_steps):
        env.render()
        #for now, choose random action, but should extract from policy 
        action = env.action_space.sample()
        #collect data for time step
        observation, reward, done, info = env.step(action)
        #prints when the episode is finished
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if t == num_steps - 1:
            print("Episode finished after {} timesteps".format(t+1))
env.close()

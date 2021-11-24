import gym
env = gym.make('BipedalWalker-v3')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if t == 99:
            print("Episode finished after {} timesteps".format(t+1))
env.close()

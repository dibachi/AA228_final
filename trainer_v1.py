from tensorforce.agents import Agent
from tensorforce.environments import Environment
import numpy as np
import time
import csv
#definitely not the best way to do this but I don't care
action_decode = np.zeros((81,4))
count = 0
vals = range(-1,2) # gets -1, 0, and 1 (2 is excluded in python)
for i in vals:
        for j in vals:
            for k in vals:
                for l in vals:
                    action_decode[count] = [i, j, k, l]
                    count = count+1

# #reward writing
# def reward_write(reward):
#     with open('reward_dqn.csv', mode='w') as f:

#         writer = csv.writer(f, lineterminator = '\n')
#         writer.writerow([episode, reward])
# #state writing
# def state_write(episode, states):
#     with open('states_dqn.csv', mode='w') as f:
#         writer = csv.writer(f, lineterminator = '\n')
#         writer.writerow([episode, states])
# #action writing
# def act_write(episode, actions):
#     with open('actions_dqn.csv', mode='w') as f:
#         writer = csv.writer(f, lineterminator = '\n')
#         writer.writerow([episode, actions])


num_episodes = 200
max_timesteps = 1600

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=max_timesteps
) 
Q_network = [
            dict(type='dense', size=170, activation='tanh'),
            dict(type='dense', size=170, activation='tanh')
        ]

#implementing epsilon-greedy w/ decay
# alpha = 1#0.9999
# epsilon = 0.1

agent = Agent.create(
    agent='dqn',
    states=environment.states(),
    actions=dict(type="int", shape=(), num_values=81),
    max_episode_timesteps=max_timesteps, 
    memory=10000,
    batch_size=16,
    learning_rate=1e-3,
    network=Q_network,
    discount=0.99,
    horizon=int(max_timesteps/10),
    exploration=0.1
)
render_freq = 10
print_freq = 1
reward_per_episode = np.zeros(num_episodes)

for episode in range(num_episodes):
    # st = time.time()
    # Initialize episode
    states = environment.reset()
    terminal = False
    # epsilon = alpha*epsilon
    # sum_reward = 0
    if episode%print_freq == 0:
        print(episode)
        # print(agent.get_specification()['exploration'])
    while not terminal:
        # Episode timestep
        if episode%render_freq == 0:
            environment.environment.render()

        actions = action_decode[agent.act(states=states)]
        # if np.random.random_sample() < epsilon:
            # actions = action_decode[np.random.randint(0,81)]
            # print("Did a random action! {epsilon}")
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        # sum_reward += reward
        state_write(episode, states)
        act_write(episode, actions)
        reward_write(episode, reward)
    
    
    # en = time.time()
    # print(en-st)
    # reward_per_episode[episode] = sum_reward
    # print(sum_reward)

agent.close()
environment.close()

# print(np.sum(reward_per_episode)/num_episodes)

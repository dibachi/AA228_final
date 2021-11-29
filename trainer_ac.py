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

num_episodes = 1000
max_timesteps = 1600

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=max_timesteps
) 
Q_network = [
            dict(type='dense', size=170, activation='tanh'),
            dict(type='dense', size=170, activation='tanh')
        ]

agent = Agent.create(
    agent='ac',
    states=environment.states(),
    actions=dict(type="int", shape=(), num_values=81),
    max_episode_timesteps=max_timesteps, 
    memory=10000,
    batch_size=64,
    learning_rate=1e-3,
    network=Q_network,
    discount=0.99,
    horizon=1,
    exploration=0.5
)
render_freq = 10
print_freq = 1
reward_per_episode = np.zeros(num_episodes)

for episode in range(num_episodes):
    # Initialize episode
    states = environment.reset()
    terminal = False
    internals = agent.initial_internals()
    episode_states = list()
    episode_internals = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()
    sum_rewards = 0.0
    while not terminal:
        episode_states.append(states)
        episode_internals.append(internals)
        action_ind, internals = agent.act(states=states, internals=internals, independent=True)
        actions = action_decode[action_ind]
        episode_actions.append(action_ind)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)
        sum_rewards += reward
    print('Episode {}: {}'.format(episode, sum_rewards))

    # Feed recorded experience to agent
    agent.experience(
        states=episode_states, internals=episode_internals, actions=episode_actions,
        terminal=episode_terminal, reward=episode_reward
    )

    # Perform update
    agent.update()
    

sum_rewards = 0.0
for _ in range(num_episodes):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        environment.environment.render()
        action_ind, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=False
        )
        actions = action_decode[action_ind]
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward


agent.close()
environment.close()


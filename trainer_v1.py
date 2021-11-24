# from tensorforce import Agent, Environment, explorations
from tensorforce.agents import Agent
from tensorforce.environments import Environment
# from tensorforce.core import EpsilonDecay
import numpy as np
import time
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

max_timesteps = 1000

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=max_timesteps
) 
Q_network = [
            dict(type='dense', size=170, activation='tanh'),
            dict(type='dense', size=170, activation='tanh')
        ]

# EpsilonDecay(
#     initial_epsilon=0.1, 
#     final_epsilon=0.001, 
#     timesteps=max_timesteps
# )
alpha = 1#0.9999
epsilon = 0.1

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
    horizon=100
)
# print(agent.get_architecture())
# print(agent.get_specification()['exploration'])

render_freq = 10
print_freq = 1
for episode in range(1000):
    st = time.time()
    # Initialize episode
    states = environment.reset()
    terminal = False
    epsilon = alpha*epsilon
    # agent.get_specification()['exploration'] = epsilon
    #I really hope I can find a better way to do this
    # agent = Agent.create(
    #     agent='dqn',
    #     states=environment.states(),
    #     actions=dict(type="int", shape=(), num_values=81),
    #     max_episode_timesteps=max_timesteps, 
    #     memory=10000,
    #     batch_size=16,
    #     learning_rate=1e-3,
    #     network=Q_network,
    #     discount=0.99,
    #     horizon=100,
    #     exploration=epsilon
    # )
    
    if episode%print_freq == 0:
        print(episode)
        # print(agent.get_specification()['exploration'])
    while not terminal:
        # Episode timestep
        if episode%render_freq == 0:
            environment.environment.render()

        actions = action_decode[agent.act(states=states)]
        if np.random.random_sample() < epsilon:
            actions = action_decode[np.random.randint(0,81)]
            # print("Did a random action! {epsilon}")
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        # if episode%render_freq == 0:
        #     print('actions')
        #     print(actions)
    en = time.time()
    print(en-st)

agent.close()
environment.close()

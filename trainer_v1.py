from tensorforce import Agent, Environment
import numpy as np

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

environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=300
) 

agent = Agent.create(
    agent='dqn',
    states=environment.states(),
    actions=dict(type="int", shape=(), num_values=81),
    max_episode_timesteps=300, 
    memory=10000,
    batch_size=64,
    learning_rate=1e-3,
    network='auto',
    discount=0.99,
    horizon=100,
    exploration=0.1
)

render_freq = 10
print_freq = 1
for episode in range(1000):

    # Initialize episode
    states = environment.reset()
    terminal = False
    if episode%print_freq == 0:
        print(episode)
    while not terminal:
        # Episode timestep
        if episode%render_freq == 0:
            environment.environment.render()
        actions = action_decode[agent.act(states=states)]
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        # if episode%render_freq == 0:
        #     print('actions')
        #     print(actions)

agent.close()
environment.close()

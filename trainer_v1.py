from tensorforce import Agent, Environment

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='BipedalWalker-v3', max_episode_timesteps=300
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=100),
    exploration=0.7
)
# policy=dict(type='parameterized_action_value', network='auto'),
# agent = Agent.create(
#     agent='dqn', 
#     environment=environment, 
#     memory=10000, 
#     batch_size=64,
#     learning_rate=1e-3,
#     discount=0.95,
#     exploration=0.1,
#     # update=dict(unit='timesteps', batch_size=64),
#     # optimizer=dict(type='adam', learning_rate=3e-4),
#     # objective='policy_gradient',
#     # reward_estimation=dict(horizon=20)
# )

# Train for 300 episodes
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
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()

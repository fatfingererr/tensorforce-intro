from tensorforce.agents import PPOAgent

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=dict(type='float', shape=(10,)),
    actions_spec=dict(type='int', num_actions=10),
    network_spec=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batch_size=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

# Get new data from somewhere, e.g. a client to a web app
client = MyClient('http://127.0.0.1', 8080)

# Poll new state from client
state = client.get_state()

# Get prediction from agent, execute
action = agent.act(state)
reward = client.execute(action)

# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)

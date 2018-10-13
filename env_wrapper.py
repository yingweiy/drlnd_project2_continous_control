from unityagents import UnityEnvironment

class EnvironmentWrapper:
    def __init__(self, fn='Reacher_Linux_20Agents/Reacher.x86_64'):
        self.env = UnityEnvironment(file_name=fn)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        states = self.reset()
        self.state_size = states.shape[1]
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('Each observes a state with length: {}'.format(self.state_size))

    def render(self):
        self.env.render()

    def reset(self):
        # reset the environment
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        # number of agents
        self.num_agents = len(self.env_info.agents)
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        # examine the state space
        states = self.env_info.vector_observations
        return states

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        return next_states, rewards, dones, env_info

    def close(self):
        self.env.close()



from unityagents import UnityEnvironment
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class EnvironmentWrapper:
    def __init__(self, fn='Reacher_Linux_SingleAgent/Reacher.x86_64'):
        self.env = UnityEnvironment(file_name=fn)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.scaler = joblib.load('state_scaler.pkl')
        state = self.reset()
        self.state_size = state.shape[0]
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('Each observes a state with length: {}'.format(self.state_size))

    #def normalize(self, state):
    #    #return self.scaler.transform([state])[0]
    #    return state

    def reset(self):
        # reset the environment
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        # number of agents
        self.num_agents = len(self.env_info.agents)
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        # examine the state space
        state = self.env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]  # send all actions to tne environment
        next_state = env_info.vector_observations[0]  # get next state (for each agent)
        reward = env_info.rewards[0]  # get reward (for each agent)
        done = env_info.local_done[0]  # see if episode finished
        return next_state, reward, done, env_info

    def close(self):
        self.env.close()



from env_wrapper import EnvironmentWrapper
from ddpg_agent import Agent
import torch
from collections import deque
import numpy as np

class DDPG:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.score_window = 100

    def train(self, n_episodes=300, max_t=1000, verbose=True):
        scores_deque = deque(maxlen=self.score_window)
        scores = []
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}\tDone in {} steps.'.format(i_episode, np.mean(scores_deque), t), end="")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if verbose:
                if i_episode % self.score_window == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return scores


if __name__=='__main__':
    env = EnvironmentWrapper()
    agent = Agent(state_size=env.state_size, action_size=env.action_size, random_seed=42)
    DRL = DDPG(env, agent)
    DRL.train()




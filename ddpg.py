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

    def train(self, n_episodes=500, max_t=2000, target_score=30.0):
        scores_deque = deque(maxlen=self.score_window)
        scores_global = []
        for i_episode in range(1, n_episodes + 1):
            states = env.reset()
            scores = np.zeros(self.env.num_agents)
            agent.reset()

            for t in range(max_t):
                actions = agent.act(states)
                next_states, rewards, dones, _ = env.step(actions)
                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                scores += rewards
                if np.any(dones):
                    break
            score = np.mean(scores)
            scores_deque.append(score)
            avg_score = np.mean(scores_deque)
            scores_global.append(scores)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score), end="")
            if avg_score>target_score:
                print('Reached target score {} in {} episodes.'.format(target_score, i_episode))
                break
            if i_episode % self.score_window == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        return scores_global


if __name__=='__main__':
    env = EnvironmentWrapper()
    agent = Agent(state_size=env.state_size, action_size=env.action_size, random_seed=42)
    DRL = DDPG(env, agent)
    DRL.train()




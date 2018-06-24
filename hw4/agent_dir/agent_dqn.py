from agent_dir.agent import Agent

import sys
import numpy as np
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cur = 0
        self.num = 0
        self.obs = np.empty([self.capacity] + [84, 84, 4], dtype=np.float32)
        self.actions = np.empty([self.capacity], dtype=np.int32)
        self.rewards = np.empty([self.capacity], dtype=np.float32)
        self.done = np.empty([self.capacity], dtype=np.bool)

    def push(self, obs, action, reward, done):
        self.obs[self.cur] = obs
        self.actions[self.cur] = action
        self.rewards[self.cur] = reward
        self.done[self.cur] = done
        self.cur = (self.cur + 1) % self.capacity
        self.num = min((self.num + 1), self.capacity)
        
    def sample(self, num):
        ids = np.array([random.randint(0, self.num - 2) for i in range(num)])
        obs_batch = self.obs[ids]
        act_batch = self.actions[ids]
        reward_batch = self.rewards[ids]
        next_obs_batch = self.obs[(ids + 1) % self.capacity]
        done_batch = np.array([1.0 if e else 0.0 for e in self.done[ids]])
        return obs_batch, act_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return self.num

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        self.use_cuda = torch.cuda.is_available()
        self.env = env
        self.replay_buffer_size = 10000
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 1.5e-4
        self.num_actions = env.action_space.n
        self.eps_end = 0.025
        self.eps_start = 1.0
        self.eps_decay = 2000
        self.i_episode = 0
        self.update_times = 0
        self.update_freq = 4
        self.target_update_freq = 1000

        self.Q = DQN(num_actions=self.num_actions)
        self.target_Q = DQN(num_actions=self.num_actions)
        if self.use_cuda:
            self.Q = self.Q.cuda()
            self.target_Q = self.target_Q.cuda()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayMemory(self.replay_buffer_size)

        if args.test_dqn:
            #you can load your model here
            self.Q.load_state_dict(torch.load('dqn_params.pt'))
            print('loading trained model')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        total_reward = 0
        total_reward_log = []
        last_obs = self.env.reset()
        for t in range(10000000):
            action = self.make_action(np.array(last_obs), test=False)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            reward = np.clip(reward, 0, 1)
            self.replay_buffer.push(last_obs, action, reward, done)
            if done:
                total_reward_log.append(total_reward)
                total_reward = 0
                self.i_episode += 1
                obs = self.env.reset()
            last_obs = obs

            if len(self.replay_buffer) > 256 and t % self.update_freq == 0:
                self.update_times += 1
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.replay_buffer.sample(self.batch_size)

                obs_batch = Variable(torch.from_numpy(np.array(obs_batch)))
                act_batch = Variable(torch.from_numpy(np.array(act_batch)).long())
                rew_batch = Variable(torch.from_numpy(np.array(rew_batch)))
                next_obs_batch = Variable(torch.from_numpy(np.array(next_obs_batch)))

                not_done_mask = Variable(torch.from_numpy(1 - done_batch)).type(torch.FloatTensor)
                if self.use_cuda:
                    act_batch = act_batch.cuda()
                    obs_batch = obs_batch.cuda()
                    rew_batch = rew_batch.cuda()
                    next_obs_batch = next_obs_batch.cuda()
                    not_done_mask = not_done_mask.cuda()
                #compute Q values
                current_Q_values = self.Q(obs_batch).gather(1, act_batch.unsqueeze(1))
                next_max_q = self.target_Q(next_obs_batch).detach().max(1)[0]
                next_Q_values = not_done_mask * next_max_q
                target_Q_values = rew_batch + (self.gamma * next_Q_values)
                
                loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                if self.update_times % self.target_update_freq == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())

            #if self.update_times % 5000 == 0 and self.update_times > 0:
            #    mean_total_reward = sum(total_reward_log) / float(len(total_reward_log))
            #    print('Timestep %d: mean reward %5f' % (self.update_times, mean_total_reward))
            #    sys.stdout.flush()
            #    torch.save(self.Q.state_dict(), 'dqn_params.pt')

            if len(total_reward_log) >= 100:
                mean_total_reward = sum(total_reward_log) / float(len(total_reward_log))
                max_reward = np.max(total_reward_log)
                print('Ep %d Up %d: mean reward %.2f, max_reward %.1f' % (self.i_episode, self.update_times, mean_total_reward, max_reward))
                sys.stdout.flush()
                total_reward_log = []
                torch.save(self.Q.state_dict(), 'dqn_params.pt')


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        sample = random.random()
        eps_threshold = max(self.eps_start - self.i_episode / self.eps_decay, self.eps_end)
        if test:
            eps_threshold = self.eps_end
        if sample > eps_threshold:
            obs = Variable(torch.from_numpy(observation).type(torch.FloatTensor).unsqueeze(0))
            if self.use_cuda:
                obs = obs.cuda()
            return self.Q(obs).data.max(1)[1].cpu()[0]
        else:
            return random.randrange(self.num_actions)


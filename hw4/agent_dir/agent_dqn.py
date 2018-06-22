from agent_dir.agent import Agent

import sys
import numpy as np
from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, *args):
        self.memory.append(None)
        self.memory[len(self.memory) - 1] = Transition(*args)
        self.memory = self.memory[-self.capacity:]
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
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
        self.learning_rate = 0.00015
        self.num_actions = env.action_space.n
        self.eps_end = 0.01
        self.eps_start = 1.0
        self.eps_decay = 100000
        self.update_times = 0
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
        i_episode = 0
        total_reward = 0
        total_reward_log = []
        last_obs = self.env.reset()
        for t in count():
            if self.update_times == 1e7:
                break
            action = self.make_action(last_obs, test=False)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.replay_buffer.push(last_obs, action, obs, reward, done)
            if done:
                total_reward_log.append(total_reward)
                total_reward_log = total_reward_log[-100:]
                total_reward = 0
                i_episode += 1
                obs = self.env.reset()
            last_obs = obs

            if len(self.replay_buffer) > 1000:
                self.update_times += 1
                transitions = self.replay_buffer.sample(self.batch_size)
                transitions = Transition(*zip(*transitions))
                obs_batch = Variable(torch.from_numpy(np.array(transitions.obs)).type(torch.FloatTensor))
                act_batch = Variable(torch.from_numpy(np.array(transitions.action)).long())
                rew_batch = Variable(torch.from_numpy(np.array(transitions.reward))).type(torch.FloatTensor)
                next_obs_batch = Variable(torch.from_numpy(np.array(transitions.next_obs)).type(torch.FloatTensor))

                not_done_mask = Variable(torch.from_numpy(1 - np.array(transitions.reward))).type(torch.FloatTensor)
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

            if self.update_times % 5000 == 0 and self.update_times > 0:
                mean_total_reward = sum(total_reward_log) / float(len(total_reward_log))
                print('Timestep %d: mean reward %5f' % (self.update_times, mean_total_reward))
                sys.stdout.flush()
                torch.save(self.Q.state_dict(), 'dqn_params.pt')

            if i_episode > 0 and i_episode % 100 == 0 and done == 1:
                mean_total_reward = sum(total_reward_log) / float(len(total_reward_log))
                max_reward = np.max(total_reward_log)
                print('Ep %d: mean reward %3f, max_reward %3f' % (i_episode, mean_total_reward, max_reward))
                sys.stdout.flush()


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
        if not test:
            sample = random.random()
            #d = 0 if self.update_times > self.eps_decay else (self.eps_start - self.eps_end) * (1 - self.update_times / self.eps_decay)
            #eps_threshold = self.eps_end + d
            eps_threshold = self.eps_end
            if sample > eps_threshold:
                obs = Variable(torch.from_numpy(observation).type(torch.FloatTensor).unsqueeze(0))
                if self.use_cuda:
                    obs = obs.cuda()
                return self.Q(obs).data.max(1)[1].cpu()[0]
            else:
                return random.randrange(self.num_actions)
        else:
            obs = Variable(torch.from_numpy(observation).type(torch.FloatTensor).unsqueeze(0))
            if self.use_cuda:
                obs = obs.cuda()
                return self.Q(obs).cpu().data.max(1)[1].numpy()[0]


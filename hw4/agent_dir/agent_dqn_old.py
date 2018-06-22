from agent_dir.agent import Agent
import numpy as np

import sys
import random
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def myprepro(I):
    ''' origin shape: (84, 84, 4) '''
    I = I[13:,5:79,:]
    I = np.sum(I, axis=2)
    I[I!=0] = 1
    ''' after shape: (71, 74, 1) '''
    return I.astype(np.float).ravel()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(71 * 74, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, 3)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.out(x)
        return out

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        self.LEARNING_RATE = 1e-2
        self.BATCH_SIZE = 256
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.use_cuda = torch.cuda.is_available()
        self.target_net = DQN()
        self.eval_net = DQN()
        self.memory = ReplayMemory(10000)

        self.update_times = 0

        if args.test_dqn:
            #you can load your model here
            self.eval_net.load_state_dict(torch.load('dqn_params.pt'))
            print('loading trained model')

        if self.use_cuda:
            self.target_net = self.target_net.cuda()
            self.eval_net = self.eval_net.cuda()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LEARNING_RATE)
        self.env = env

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    
    def optim_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        if self.use_cuda:
            non_final_next_states = non_final_next_states.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
        state_action_values = self.eval_net(state_batch).gather(1, action_batch)
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(torch.Tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).cpu().max(1)[0] if self.use_cuda else self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        if self.use_cuda:
            expected_state_action_values = expected_state_action_values.cuda()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.eval_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_times += 1
        if self.update_times % 500 == 0:
            print('Batch update %d, actual rewards: %5f, expected rewards: %5f' % (self.update_times, np.mean(state_action_values.cpu().data.numpy()), np.mean(expected_state_action_values.cpu().data.numpy())))
            sys.stdout.flush()
        if self.update_times % 5000 == 0:
            torch.save(self.eval_net.state_dict(), 'dqn_params.pt')

    def train(self):
        """
        Implement your training algorithm here
        """
        #running_reward = None
        #reward_sum = 0
        i_episode = 0
        while self.update_times < 2e8:
            i_episode += 1
            state = self.env.reset()
            state = myprepro(state)
            state = torch.from_numpy(state).float().unsqueeze(0)
            done = False
            while not done:
                action = self.make_action(state, test=False)
                state_, reward, done, _ = self.env.step(action)
                action = torch.LongTensor([[action - 1]])
                #reward_sum += reward
                state_ = myprepro(state_)
                state_ = torch.from_numpy(state_).float().unsqueeze(0)
                reward = torch.Tensor([reward])
                self.memory.push(state, action, state_, reward)
                self.optim_model()
                state = state_

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            
            #if done:
            #    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            #    print('ep %d: resetting env. episode reward total was %f. running mean: %f' % (i_episode, reward_sum, running_reward))
            #    reward_sum = 0

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
            state = observation
            action = None
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.update_times / self.EPS_DECAY)
            if sample > eps_threshold:
                state = Variable(state)
                if self.use_cuda:
                    state = state.cuda()
                probs = self.eval_net(state)
                action = torch.max(probs, 1)[1].cpu().data[0]
            else:
                action = np.random.randint(0, 3)
        
            return action + 1
        else:
            state = myprepro(observation)
            state = torch.from_numpy(state).float().unsqueeze(0)
            state = Variable(state)
            if self.use_cuda:
                state = state.cuda()
            probs = self.eval_net(state)
            m = Categorical(probs)
            action = m.sample()
            return action.data[0] + 1

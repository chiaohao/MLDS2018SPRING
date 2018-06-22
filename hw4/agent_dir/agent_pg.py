from agent_dir.agent import Agent
import scipy
import numpy as np

import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def myprepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float).ravel()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 256)
        self.affine2 = nn.Linear(256, 3)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.LEARNING_RATE = 1e-4
        self.BATCH_SIZE = 1
        self.MINI_BATCH_SIZE = 256
        self.GAMMA = 0.99

        self.use_cuda = torch.cuda.is_available()
        self.policy = Policy()

        if args.test_pg:
            #you can load your model here
            self.policy.load_state_dict(torch.load('pg_params.pt'))
            print('loading trained model')

        if self.use_cuda:
            self.policy = self.policy.cuda()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.LEARNING_RATE)
        

        self.env = env
        #seed = random.randint(0, 100000)
        #self.env.seed(seed)
        #torch.manual_seed(seed)

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
        running_reward = None
        reward_sum = 0
        for i_episode in range(50000):
            sys.stdout.flush()
            state = self.env.reset()
            for t in range(10000):
                action = self.make_action(state, test=False)
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                
                self.policy.rewards.append(reward)
                if done:
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('ep %d: resetting env. episode reward total was %f. running mean: %f' % (i_episode, reward_sum, running_reward))
                    reward_sum = 0
                    break
            if i_episode % self.BATCH_SIZE == 0:
                self.finish_episode()
            if i_episode % 50 == 49:
                #print('ep %d: model saving...' % (i_episode))
                torch.save(self.policy.state_dict(), 'pg_params.pt')

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
                ***myprepro -> (6400,)

        Return:
            action: int
                the predicted action from trained model
        """
        state = myprepro(observation)
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = Variable(state)
        if self.use_cuda:
            state = state.cuda()
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        if not test:
            self.policy.saved_log_probs.append(m.log_prob(action))
        return action.data[0] + 1
        #return self.env.get_random_action()
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.GAMMA * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        batch_num = len(policy_loss) // self.MINI_BATCH_SIZE + 1 if len(policy_loss) % self.MINI_BATCH_SIZE != 0 else len(policy_loss) // self.MINI_BATCH_SIZE
        for i in range(batch_num):
            self.optimizer.zero_grad()
            _policy_loss = torch.cat(policy_loss[i * self.MINI_BATCH_SIZE:(i + 1) * self.MINI_BATCH_SIZE]).sum()
            _policy_loss.backward()
            self.optimizer.step()
        
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.linalg

class LQREnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,S_dim=5,A_dim=3):
        self.S_dim = S_dim
        self.A_dim = A_dim
        #Q1 = np.random.randn(S_dim,S_dim)
        #R1 = np.random.randn(A_dim,A_dim)
        #self.Q = np.dot(Q1,Q1.T)
        #self.R = np.dot(R1,R1.T)
        #self.Q = self.Q/scipy.linalg.eig(self.Q)[0][0]
        #self.R = self.R/scipy.linalg.eig(self.R)[0][0]
        self.Q = np.eye(S_dim)
        self.R = np.eye(A_dim)
        self.A = np.random.randn(S_dim,S_dim)
        self.B = np.random.randn(S_dim,A_dim)
        L = 10
        self.observation_space = spaces.Box(low=-L*np.ones((S_dim,)),high=L*np.ones((S_dim,)),dtype=np.float32)
        self.action_space = spaces.Box(low=-L*np.ones((A_dim,)),high=L*np.ones((A_dim,)),dtype=np.float32)
        self.T = 500
        self.t = 0

    def step(self, action):
        cost = np.inner(np.dot(self.state,self.Q),self.state) \
               + np.inner(np.dot(action,self.R),action)
        self.state = np.dot(self.A,self.state)+np.dot(self.B,action)
        self.t += 1
        return self.state,-1*cost,self.t>=self.T,{}
    def compute_opt_controller(self):
        ''' u = K x '''
        #first, try to solve the ricatti equation
        self.X = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        #compute the LQR gain
        self.K = np.matrix(scipy.linalg.inv(self.B.T*self.X*self.B+self.R)*(self.B.T*self.X*self.A))
        self.eigVals, eigVecs = scipy.linalg.eig(self.A-self.B*self.K)
        return self.K, self.X, self.eigVals
 
    def reset(self):
        self.state = np.random.randn(self.S_dim)
        return self.state
    def render(self, mode='human', close=False):
        pass

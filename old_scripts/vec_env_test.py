import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import time, os, itertools, sys
from CIL_venv.lib.rl_baselines_zoo.utils import ALGOS, create_test_env, find_saved_model
#from stable_baselines.common.tf_layers import conv_to_fc,linear
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from functools import partial
from matplotlib import animation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
import faulthandler; faulthandler.enable()

#Eager execution
#tf.compat.v1.enable_eager_execution()
#Suppress warnings


#Constants
ATARI_ENVS = ['BeamRiderNoFrameskip-v4','BreakoutNoFrameskip-v4','EnduroNoFrameskip-v4','PongNoFrameskip-v4','QbertNoFrameskip-v4','SeaquestNoFrameskip-v4','SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4','Pong-v4']
ZOO_DIR = 'CIL_venv/lib/rl_baselines_zoo/trained_agents/'
BEST_ALGO = {'BeamRiderNoFrameskip-v4':'acktr','BreakoutNoFrameskip-v4':'acktr','EnduroNoFrameskip-v4':'dqn','PongNoFrameskip-v4':'dqn','QbertNoFrameskip-v4':'ppo2',
             'SeaquestNoFrameskip-v4':'dqn','SpaceInvadersNoFrameskip-v4':'dqn','MsPacmanNoFrameskip-v4':'acer','BipedalWalker-v2':'sac','LunarLander-v2':'dqn','LunarLanderContinuous-v2':'sac',
             'CartPole-v1':'ppo2','Acrobot-v1':'ppo2'}
MAX_SCORES = {'PongNoFrameskip-v4':21,'EnduroNoFrameskip-v4':700}
EPS = 1e-9

#Environment mod methods

def make_env(env_id,n_env=1):
    env_maker = (lambda : wrap_deepmind(make_atari(env_id))) if env_id in ATARI_ENVS else (lambda : gym.make(env_id))
    if n_env==1:
        return DummyVecEnv([env_maker for i in range(n_env)])
    else:
        return SubprocVecEnv([env_maker for i in range(n_env)])
def warp(obs):
     return cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)[...,np.newaxis]
def reset_env(env,states=None):
    #Built for VecEnv
    obs = env.reset()
    if 0:
        if states[0] is not None:
            if env_id in ATARI_ENVS:
                #for ind,ale in enumerate(env.get_attr('ale')):
                #    ale.restoreState(states[ind])
                for ind,restore_state in enumerate(env.get_attr('restore_state')):
                    restore_state(states[ind])
                #print(dir(env),dir(env.unwrapped))
                obs = warp(env.unwrapped._get_obs())
            else:
                self.unwrapped.envs[0].unwrapped.state = states[0]
                if '_get_ob' in dir(self.unwrapped.envs[0].unwrapped):
                    obs = [env.unwrapped.envs[0].unwrapped._get_ob()]
                else:
                    obs = [np.array(states[0])]
    return obs
def get_state(env_id,env):
    #return [None]*20
    if env_id in ATARI_ENVS:
        if isinstance(env.unwrapped,DummyVecEnv) or isinstance(env.unwrapped,SubprocVecEnv):
            return env.env_method('clone_state')
            return [ale.cloneState() for ale in env.get_attr('ale')]
            return [env.unwrapped.envs[i].unwrapped.ale.cloneSystemState() for i in range(len(env.unwrapped.envs))]
        else:
            return env.unwrapped.ale.cloneSystemState()
    else:
        return env.unwrapped.envs[0].unwrapped.state
def de_framestack(obs):
    return obs[...,-1:]
def add_batch_dim(obs,**kwargs):
    return obs[np.newaxis,:]
def add_history(obs,env_id,**kwargs):
    pass
def append_history_to_array(obs,**kwargs):
    return obs

##########   Get experts and trajectories

def get_expert_model(env_id,algo=None):
    '''Require RL Baselines Zoo Package'''
    algo = algo or BEST_ALGO.get(env_id,'dqn')
    print(algo)
    general_model_path = ZOO_DIR+algo
    model_path = find_saved_model(algo, general_model_path, env_id, load_best=False)
    print('Loading {}'.format(env_id))
    model = ALGOS[algo].load(model_path, env=None)#make_env(env_id))
    return model

def get_trajectories(policy,env_id,N_traj=1,path=None,render=False,verbose=0,df_agg=None,df_E=None,pair_with_E=False,
                     obs_preprocess=None,T_max=None,seed=None,obs_postprocess=None,gif_path=None,init_ts=None,parallel_env=False):

    df_columns = ['obs','obs_next','state','state_next','action','action_prev','rew','t','traj_ind','weight','E_ind']

    #Load trajs_df if exists
    traj_df = df_agg or pd.DataFrame(columns=df_columns)
    n_loaded=0
    if path is not None:
        if os.path.exists(path):
            traj_df = pd.read_pickle(path)
            n_loaded = traj_df['traj_ind'].nunique()
            if n_loaded<N_traj:
                if verbose>0:
                    print('Beginning generation of {} more trajs (found {}).'.format(N_traj-n_loaded,n_loaded))
            else:
                if verbose>0:
                    print('Loaded {} trajs from {}'.format(N_traj,path))
                max_ind = traj_df.index[traj_df['traj_ind']==N_traj-1][-1]+1
                print(max_ind)
                return traj_df.iloc[:max_ind]
    
    obs_post = (lambda obs:obs) if obs_postprocess is None else obs_postprocess    
    if gif_path is not None:
        img_array = []    

    #Generate trajectories and add them to dataframe
    num_subproc = min(N_traj,20) if parallel_env else 1
    run_seed = seed or np.random.randint(10000)
    env = make_env(env_id,n_env=num_subproc)
    env.seed(run_seed)
    render = render and not parallel_env
    episode_rews = []
    for traj_num in range(n_loaded,N_traj,num_subproc):
        n_env = min(num_subproc,N_traj-traj_num)
        
        #Handle initialization of initial state and previous action
        env_state, action_prev, E_inds, t_0s, ts, T_max = [None]*n_env, [None]*n_env, [None]*n_env, [0]*n_env, np.zeros((n_env,)), T_max or 100000
        if pair_with_E and df_E is not None:
            for vec_env_ind in range(n_env):
                t_0s = init_ts[traj_num + vec_env_ind] if init_ts is not None else 0
                E_inds = df_E[df_E['t']>=t_init][df_E['traj_ind']==traj_num].index
                if len(E_inds)>0:
                    T_max = df_E['t'].loc[E_inds].max()
                    env_state,action_prev,t = df_E[['state','action_prev','t']].loc[E_inds[0]]
            
        obs = reset_env(env,env_state)
        episode_rew, done = 0,False
        while not (np.array(done).all() or (ts>=T_max).all()):
            obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
            #if verbose==4:
            #    print(obs_proc,obs_proc.shape)
            action = policy(obs_proc)
            #print(obs.shape,action.shape)
            if render:
                env.render(); #time.sleep(.02)
            if gif_path is not None:
                if env_id in ATARI_ENVS:
                    pic = env.env_method('_get_image')[0]
                else:
                    pic = env.env_method('render',mode='rgb')
                img_array.append(pic)
            obs_next, rew, done, _ = env.step(action)
            env_state_next = get_state(env_id,env)
            for vec_env_ind in range(n_env): #Only save until you have N_traj total
                traj_df.loc[len(traj_df)] = pd.Series({'obs':obs_post(obs)[vec_env_ind],'obs_next':obs_post(obs_next)[vec_env_ind],'state':env_state[vec_env_ind],
                                               'state_next':env_state_next[vec_env_ind],'action':action[vec_env_ind],'action_prev':action_prev[vec_env_ind],
                                               'rew':rew[vec_env_ind],'t':ts[vec_env_ind],'traj_ind':traj_num+vec_env_ind,'weight':1.0,
                                               'E_ind':len(traj_df) if E_inds[vec_env_ind] is None else E_inds[vec_env_ind][ts[vec_env_ind]-t_0s[vec_env_ind]]}) #TODO fix E_ind
            env_state,obs,action_prev = env_state_next,obs_next,action
            ts += 1
            episode_rew += rew
        episode_rews.extend(episode_rew)
        if verbose>1:
            print("Episode(s) {} reward: {}, lens {}".format([traj_num+i for i in range(n_env)],episode_rew,ts))
    print("Avg reward {:.1f}".format(np.mean(episode_rews)))
    if render:
        env.close()
    if gif_path is not None:
        make_gif(img_array,gif_path)
    if path is not None:
        if os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        traj_df.to_pickle(path)
        if verbose>0:
            print('Saved {} trajs to {}'.format(N_traj,path))

    return traj_df
if __name__=='__main__':
    env_id = 'PongNoFrameskip-v4'
    #model_E = get_expert_model(env_id)#,algo='a2c')
    #pi_E = lambda obs: model_E.predict(obs)[0] #None
    pi_bad = lambda obs: np.random.randint(6,size=obs.shape[0])
    #env = make_atari_env(env_id, num_env=4, seed=0, use_subprocess=True)
    #print('************Jonathan can figure it out too!')
    start = time.time()
    traj_df = get_trajectories(pi_bad,env_id,N_traj=20,path=None,render=False,verbose=2,T_max=1500,parallel_env=True)
    print(time.time()-start)
    print(len(traj_df))


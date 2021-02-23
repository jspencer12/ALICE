import numpy as np
import pandas as pd
import argparse
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.precision = 2
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) #hacky to get tf quiet
warnings.filterwarnings('ignore',category=UserWarning) #hacky to get tf quiet
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import gym
import time, os, itertools, sys, pickle, yaml, subprocess, copy, portalocker
from scipy.special import softmax
from cycler import cycler
from stable_baselines.common.tf_layers import conv_to_fc,linear
from stable_baselines.common.tf_util import make_session
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv, DummyVecEnv
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,QuantileTransformer,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.manifold import TSNE
from collections import deque
from densratio import densratio
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(THIS_DIR+'/rl_baselines_zoo')
from utils import ALGOS,create_test_env, find_saved_model, make_env,get_wrapper_class, get_latest_run_id,get_saved_hyperparams

#Eager execution
tf.compat.v1.enable_eager_execution()
#Suppress warnings
gym.logger.set_level(40)

#Constants
ATARI_ENVS = ['BeamRiderNoFrameskip-v4','BreakoutNoFrameskip-v4','EnduroNoFrameskip-v4','PongNoFrameskip-v4','QbertNoFrameskip-v4','SeaquestNoFrameskip-v4','SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4']
MUJOCO_ENVS = ['Hopper-v2','HalfCheetah-v2','Walker2d-v2','Ant-v2','Humanoid-v2','Reacher-v2','Hopper-v3','HalfCheetah-v3','Walker2d-v3','Ant-v3','Humanoid-v3','Reacher-v3']
ZOO_DIR = os.path.join(THIS_DIR+'/rl_baselines_zoo/trained_agents')
DISCRETE_COLORMAPS = ['Pastel1','Pastel2','Paired','Accent','Dark2','Set1','Set2','Set3','tab10','tab20','tab20b','tab20c']
BEST_ALGO = {'BeamRiderNoFrameskip-v4':'acktr','BreakoutNoFrameskip-v4':'acktr','EnduroNoFrameskip-v4':'dqn','PongNoFrameskip-v4':'dqn','QbertNoFrameskip-v4':'ppo2',
             'SeaquestNoFrameskip-v4':'dqn','SpaceInvadersNoFrameskip-v4':'dqn','MsPacmanNoFrameskip-v4':'acer','BipedalWalker-v2':'sac','LunarLander-v2':'dqn','LunarLanderContinuous-v2':'sac',
             'CartPole-v1':'dqn','MountainCar-v0':'dqn','Acrobot-v1':'dqn','MountainCar-v0':'dqn','HalfCheetah-v2':'sac','Hopper-v2':'sac','Ant-v2':'sac','Humanoid-v2':'sac','Reacher-v2':'sac','Walker2d-v2':'sac'}
MAX_SCORES = {'PongNoFrameskip-v4':21,'EnduroNoFrameskip-v4':700}
EPS = 1e-9

#Environment mod methods

def make_env(env_id,vec_env=False,norm_obs=False,norm_reward=False,norm_path=None,hyperparams=None):
    if vec_env:
        return create_test_env(env_id, is_atari=(env_id in ATARI_ENVS), log_dir=None,hyperparams=hyperparams) #Use this because it does nice preprocessing
    if env_id in ATARI_ENVS:
        #return ClipRewardEnv(WarpFrame(NoopResetEnv(gym.make(env_id), noop_max=30)))
        #return ClipRewardEnv(WarpFrame(MaxAndSkipEnv(NoopResetEnv(gym.make(env_id), noop_max=30),skip=4)))
        #Applies following wrappers NoopResetEnv(env, noop_max=30) MaxAndSkipEnv(env, skip=4) EpisodicLifeEnv(env) FireResetEnv(env) WarpFrame(env) ClipRewardEnv(env)
        return wrap_deepmind(make_atari(env_id)) 
    else:
        if (norm_obs or norm_reward or (norm_path is not None)):
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env =  VecNormalize(DummyVecEnv([gym.make(env_id)]),norm_obs=norm_obs, norm_reward=norm_reward,training=False) #Normalizes observations with a 
            if norm_path is not None:
                if os.path.exists(norm_path):
                    env = VecNormalize.load(norm_path, env)
                    # Deactivate training and reward normalization
                    env.training = False
                    env.norm_reward = False
        else:
            return gym.make(env_id)
        #return create_test_env(env_id)
#        return gym.make(env_id)

def reshape_recolor_atari(obs):
     return cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)[:,:,None]
def reset_env(env,state=None,state_prev=None,action_prev=None):
    vecenv = hasattr(env,'envs')
    env_id = env.envs[0].unwrapped.spec.id if vecenv else env.unwrapped.spec.id
    #print(env_id,'Reset')
    if env_id in ATARI_ENVS:
        if vecenv:
            return env.reset()
        env.unwrapped.reset()
        env.reset()
        obs = reshape_recolor_atari(env.unwrapped._get_obs())
        if state is not None:
            env.unwrapped.restore_full_state(state)
        if state_prev is not None:
            env.unwrapped.restore_full_state(state_prev)
        action_prev = action_prev or 0 #action_prev=0 is no-op
        #print(action_prev)
        obs = env.step(int(action_prev))[0]
    else:
        obs = env.reset()
        if state is not None:
            env.unwrapped.state = state
            obs = [np.array(state)]
        #else:
        #    self.unwrapped.envs[0].unwrapped.state = state
        #    if '_get_ob' in dir(self.unwrapped.envs[0].unwrapped):
        #        obs = [env.unwrapped.envs[0].unwrapped._get_ob()]
        #    else:
                
    return obs
def get_state(env):
    env_id = env.envs[0].unwrapped.spec.id if hasattr(env,'envs') else env.unwrapped.spec.id
    if env_id in ATARI_ENVS:
        if isinstance(env.unwrapped,DummyVecEnv):
            return env.unwrapped.envs[0].unwrapped.clone_full_state()
        else:
            return env.unwrapped.clone_full_state()
    elif env_id in MUJOCO_ENVS:
        if isinstance(env.unwrapped,DummyVecEnv):
            return env.unwrapped.envs[0].sim.get_state()
        else:
            return env.sim.get_state()
    else:
        if isinstance(env.unwrapped,DummyVecEnv):
            return env.unwrapped.envs[0].unwrapped.state
        else:
            return env.unwrapped.state
def de_framestack(obs,*args,**kwargs):
    return obs[...,-1:]
def add_batch_dim(obs,*args,**kwargs):
    return np.array(obs)[np.newaxis,:]

def warp_obs(obs,env_id,action_prev=None,add_history=True,kill_feats=None,**kw):
    if env_id in ATARI_ENVS:
        if add_history:
            return add_history_dimension(obs,action_prev,env_id)
            #return add_history_to_pixels(obs,action_prev,env_id)
        else:
            return obs
    else:
        if add_history:
            obs = append_history_to_array(obs,action_prev,env_id,kw.get('A_shape',(1,)))
        if kill_feats is not None:
            obs = np.delete(obs,kill_feats)
        return obs

def add_history_to_pixels(obs,action,env_id):
    loc = {'PongNoFrameskip-v4':(0,12,0,12)}.get(env_id,(0,12,0,12))
    action = action or 0
    if len(obs.shape)==3: #No batch dimension
        obs[loc[0]:loc[1],loc[2]:loc[3],:] = np.array(action)*42 #255/6
    if len(obs.shape)==4:
        obs[:,loc[0]:loc[1],loc[2]:loc[3],:] = np.reshape(action,(-1,1,1,1))
    return obs
def add_history_dimension(obs,action,env_id):
    if (action is None) or (np.isnan(action).any()):
        action = 0
    new_channel = np.ones(obs.shape[:-1])*np.array(action)*42 #chop off channel dimension
    return np.stack([*np.rollaxis(obs,-1),new_channel],axis=-1)

def append_history_to_array(obs,action,env_id,A_shape=(1,)):
    if (action is None) or (np.isnan(action).any()):
        action = np.zeros(A_shape)
    #action = action or 0
    #action = 0 if np.isnan(action) else action
    #if action not in [0,1,2,1.0,2.0]:
    #    print(action)
    #print(obs,action)
    #if hasattr(action,'len'):
    #    action = action[0]
    newobs = np.hstack([np.squeeze(obs),np.squeeze(action)]).astype(np.float32) #*np.eye(6)[action]
    return newobs

##########   Get experts and trajectories
def train_zoo_agent(env_id,RL_algo='dqn',RL_expert_folder='my_RL_experts',n_steps=None,resume=False,env_kwargs=None):
    '''Require RL Baselines Zoo Package installed in same directory'''
    #assert os.path.exists('rl_baselines_zoo')
    start = time.time()
    #parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    #parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    #parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',default='', type=str)
    #parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    #parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    #parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',default=10000, type=int)
    #parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',default=5, type=int)
    #parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',default=-1, type=int)
    #parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    #parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    #parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    #parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,help='Run hyperparameters search')
    #parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    #parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,default='tpe', choices=['random', 'tpe', 'skopt'])
    #parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str, default='median', choices=['halving', 'median', 'none'])
    #parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,type=int)
    #parser.add_argument('--gym-packages', type=str, nargs='+', default=[],help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    #parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    #parser.add_argument('-uuid', '--uuid', action='store_true', default=False,help='Ensure that the run has a unique ID')
    #parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,help='Optional keyword argument to pass to the env constructor')
    if 1:
        arglist = ['python',THIS_DIR+'/rl_baselines_zoo/train.py','--algo',RL_algo,
                   '--env',env_id,'-n',str(n_steps),'-f',RL_expert_folder]
        if env_kwargs is not None:
            arglist.append('--env-kwargs')
            for k in env_kwargs:
                arglist.append(k+':'+str(env_kwargs[k]))
        print(arglist)
        subprocess.run(arglist)
        #if resume:
        #    log_path = os.path.join(ZOO_DIR, algo)
        #    load_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id)))
        #    if os.path.exists(load_path):
        #        subprocess.run(['python','rl_baselines_zoo/train.py','--algo',algo,'--env',env_id,'-n',str(n_steps),'-f',folder,'-i',load_path])
        #else:
            
    #Alternatively
    if 0:
        with open('rl_baselines_zoo/hyperparams/{}.yml'.format(algo), 'r') as f:
            hyperparams_dict = yaml.safe_load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            elif env_id in ATARI_ENVS:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))
        n_steps = n_steps or hyperparams['n_timesteps']
        print(hyperparams)
        del hyperparams['n_timesteps']
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
        params_path = "{}/{}".format(save_path, env_id)

        env = make_env(env_id,vec_env=True,hyperparams=hyperparams)
        model = ALGOS[algo](env=env, **hyperparams)
        model.learn(n_steps,verbose=1)
    print('Finished after {:.1f} minutes'.format((time.time()-start)/60))


def get_zoo_model(env_id,RL_algo=None,RL_expert_folder=None,exp_id=0,load_best=False):
    '''Require RL Baselines Zoo Package (I basically copied this from enjoy.py)

        exp_id:  (int) -1:no experiment directory (uncommon)
                        0:use most recent experiment (for self trained agents)
                        i:use particular experiment (for self trained agents)
    '''
    if RL_algo is None:
        RL_algo = BEST_ALGO[env_id]
    if RL_expert_folder is not None:
        if not os.path.exists(RL_expert_folder):
            RL_expert_folder = os.path.join(THIS_DIR,RL_expert_folder)
        if not os.path.exists(RL_expert_folder):
            raise Exception('Could not find expert directory',RL_expert_folder)
        if exp_id == 0:
            exp_id = get_latest_run_id(os.path.join(RL_expert_folder, RL_algo), env_id)
            print('Loading latest user-generated agent, id={}'.format(exp_id))
        log_path = os.path.join(RL_expert_folder, RL_algo) + ('/{}_{}'.format(env_id, exp_id) if exp_id > 0 else '/'+env_id)
    else:
        load_best = False
        log_path = os.path.join(ZOO_DIR, RL_algo)
    model_path = find_saved_model(RL_algo, log_path, env_id, load_best=load_best)
    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
    if hyperparams.get('normalize',False):
        print('Agent trained with normalized observations. Ensure identical'\
              ' normalization in future steps')
    env = make_env(env_id,vec_env=True,hyperparams=hyperparams)
    load_env = None if RL_algo == 'acer' else env
    model = ALGOS[RL_algo].load(model_path, env=load_env)
    return model
    
def eval_pi_lightweight(policy,env_id,N_traj=1,render=False,run_seed=None,vec_env=False,obs_preprocess=None,verbose=0,T_max=None):
    env = make_env(env_id,vec_env)
    np.random.seed()
    run_seed = np.random.randint(10000) if run_seed is None else run_seed
    env.seed(run_seed)
    T_max = T_max or 1000000
    episode_rews = []
    start = time.time()
    for traj_ind in range(N_traj):
        env_state, env_state_prev, action_prev, E_inds, t = None, None, None, None, 0
        obs = reset_env(env,env_state,env_state_prev,action_prev)
        obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
        episode_rew, done = 0,False
        while not (done or t>=T_max):
            obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
            action = policy(obs_proc)
            if render:
                env.render(); time.sleep(.02)
            obs_next, rew, done, _ = env.step(action)
            rew = rew[0] if vec_env else rew
            env_state_next = get_state(env)
            env_state,obs,action_prev,env_state_prev = env_state_next,obs_next,action,env_state
            t += 1
            episode_rew += rew
        episode_rews.append(episode_rew)
        if verbose>1:
            print("Episode {}, {} steps ({:.1f} steps/sec) reward: {}".format(traj_ind+1,t,t/(time.time()-ep_start_time),episode_rew))
    if verbose > 0:
        print("{} episodes, ({:.1f} min) Avg reward {:.1f}+-{:.1f},".format(N_traj,(time.time()-start)/60,np.mean(episode_rews),np.std(episode_rews)))
    env.close()
    return np.mean(episode_rews),np.std(episode_rews)
    
def FORWARD_eval_pi_lightweight(model_list,model2policy,env_id,FORWARD_H=1,N_traj=1,render=False,run_seed=None,vec_env=False,obs_preprocess=None,verbose=0,T_max=None):
    env = make_env(env_id,vec_env)
    np.random.seed()
    run_seed = np.random.randint(10000) if run_seed is None else run_seed
    env.seed(run_seed)
    
    T_max = 1000000 if T_max is None else T_max
    episode_rews = []
    start = time.time()
    for traj_ind in range(N_traj):
        env_state, env_state_prev, action_prev, E_inds, t = None, None, None, None, 0
        obs = reset_env(env,env_state,env_state_prev,action_prev)
        obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
        episode_rew, done = 0,False
        while not (done or t>T_max):
            policy = model2policy(model_list[min(t//FORWARD_H,len(model_list)-1)])
            obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
            action = policy(obs_proc)
            if render:
                env.render(); time.sleep(.02)
            obs_next, rew, done, _ = env.step(action)
            rew = rew[0] if vec_env else rew
            env_state_next = get_state(env)
            env_state,obs,action_prev,env_state_prev = env_state_next,obs_next,action,env_state
            t += 1
            episode_rew += rew
        episode_rews.append(episode_rew)
        if verbose>1:
            print("Episode {}, {} steps ({:.1f} steps/sec) reward: {}".format(traj_ind+1,t,t/(time.time()-ep_start_time),episode_rew))
    if verbose > 0:
        print("{} episodes, ({:.1f} min) Avg reward {:.1f}+-{:.1f},".format(N_traj,(time.time()-start)/60,np.mean(episode_rews),np.std(episode_rews)))
    env.close()
    return np.mean(episode_rews),np.std(episode_rews)

def get_trajectories(policy,env_id,N_traj=1,path=None,render=False,verbose=0,df_agg=None,df_init=None,init_from_df=False,
                     obs_preprocess=None,T_max=None,seed=None,obs_postprocess=None,vec_env=False,gif_path=None,init_ts=None,expert_after_n=1e10,policy_e=None,e_prepro=None,
                     randinit_t=False,horizon=None,choose_random_expert=False,init_sprime=False):

    df_columns = ['obs','obs_next','state','state_next','state_prev','action','action_prev','rew','t','traj_ind','weight_action','weight','E_ind','done']

    #Load trajs_df if exists
    n_traj_loaded = 0
    n_samp_loaded = 0
    if path is not None:
        if os.path.exists(path):
            df_saved = pd.read_pickle(path)
            n_traj_loaded = df_saved['traj_ind'].nunique()
            if n_traj_loaded<N_traj:
                if verbose>0:
                    print('Beginning generation of {} more trajs (found {}).'.format(N_traj-n_traj_loaded,n_traj_loaded))
            else:
                if verbose>0:
                    print('Loaded {} trajs from {}'.format(N_traj,path))
                max_ind = df_saved.index[df_saved['traj_ind']<N_traj][-1]+1
                return df_saved.iloc[:max_ind]
            #probably won't be both loading and agging, but if you do, agg first then load, index over both
            df_agg = df_saved if df_agg is None else pd.concat([df_agg,df_saved],ignore_index=True) 
            n_samp_loaded = len(df_agg)
    
    obs_post = (lambda obs,*a,**kw:obs) if obs_postprocess is None else obs_postprocess    
    if gif_path is not None:
        img_array = []
    #Generate trajectories and add them to dataframe
    env = make_env(env_id,vec_env)
    np.random.seed()
    run_seed = np.random.randint(10000) if seed is None else seed
    env.seed(run_seed)
    np.random.seed(seed)
    episode_rews = []
    start = time.time()
    samp_list = []
    traj_inds = np.arange(n_traj_loaded,N_traj)
    if choose_random_expert and init_from_df and df_init is not None:
        traj_inds = np.random.permutation(pd.unique(df_init['traj_ind']))[:N_traj-n_traj_loaded]
        print('random expert traj inds',traj_inds)
    for traj_ind in traj_inds:

        env_state, env_state_prev, action_prev, E_inds, t, T_max, done = None, None, None, None, 0, T_max or 100000, False

        #Handle initialization of initial state and previous action
        if init_from_df and df_init is not None:
            len_E = df_init['t'][df_init['traj_ind']==traj_ind].max()
            max_t_init = len_E if horizon is None else max(1,len_E-horizon)
            #t_init = 0 if not randinit_t else np.random.randint(max_t_init)
            if randinit_t:
                t_init = np.random.randint(max_t_init)
            elif init_ts is not None:
                t_init = init_ts[traj_ind]
            else:
                t_init = 0
            T_max = min(max_t_init if horizon is None else t_init + horizon,T_max)
            if init_sprime:
                potential_inds = df_init[df_init['traj_ind']==traj_ind][df_init['t']==t_init-1].index
                if len(potential_inds)>0:
                    ind_init = potential_inds[0]
                else:
                    continue
                env_state,env_state_prev,action_prev,t = df_init[['state_next','state','action','t']].loc[ind_init]
                t += 1
            else:
                potential_inds = df_init[df_init['traj_ind']==traj_ind][df_init['t']==t_init].index
                if len(potential_inds)>0:
                    ind_init = potential_inds[0]
                else:
                    continue
                env_state,env_state_prev,action_prev,t = df_init[['state','state_prev','action_prev','t']].loc[ind_init]

            done = df_init.loc[ind_init,'done'] if 'done' in df_init.columns else False


        obs = reset_env(env,env_state,env_state_prev,action_prev)
        env_state = get_state(env)
        obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
        #obs_deque = deque([np.zeros_like(obs_proc)]*4,maxlen=4)
        episode_rew = 0
        ep_start_time = time.time()
        while not (done or t>=T_max):
            obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
            #if verbose==4:
            #    print(obs_proc,obs_proc.shape)
            #if t>=expert_after_n:
                #obs_deque.append(obs_proc)
                #obs_plt = np.stack(obs_deque,-1)[...,-1,:]
            #    action = policy_e(obs_plt) #handles frame stacking, albeit hard to read
                #print('E',end='')
            #else:
            #    action = policy(obs_proc)
                #print(obs_proc,env.get_original_obs())
            #    obs_plt = obs
                #print('L',end='')
            action = policy(obs_proc)
            #print(action,end='')
            #if not vec_env:
            #    action = action[0]
            if render:
                env.render(); time.sleep(.02)
            #if t == -1:
            #    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            #    axs[0, 0].imshow(obs_plt[0,:,:,0]); axs[0, 1].imshow(obs_plt[0,:,:,1]); axs[1, 0].imshow(obs_plt[0,:,:,2]); axs[1, 1].imshow(obs_plt[0,:,:,3])
            #    plt.show()
            if gif_path is not None:
                pic = env.unwrapped.envs[0].unwrapped._get_image() if vec_env else env.unwrapped._get_image()
                img_array.append(pic)
            #print(action)
            obs_next, rew, done, _ = env.step(action)
            rew = rew[0] if vec_env else rew
            #print(obs,obs_post(obs))
            env_state_next = get_state(env)
            samp_list.append([obs_post(obs,env_id,action_prev=action_prev),obs_post(obs_next,env_id,action_prev=action),env_state,env_state_next,env_state_prev,action,action_prev,rew,t,traj_ind,1.0,1.0,len(samp_list)+n_samp_loaded if E_inds is None else E_inds[t-t_init],done])
            env_state,obs,action_prev,env_state_prev = env_state_next,obs_next,action,env_state
            
            t += 1
            episode_rew += rew
        episode_rews.append(episode_rew)
        if verbose>1:
            print("Episode {}, {} steps ({:.1f} steps/sec) reward: {}".format(traj_ind+1,t,t/(time.time()-ep_start_time),episode_rew))
    traj_df = pd.DataFrame(samp_list,columns=df_columns)
    if df_agg is not None:
        traj_df = pd.concat([df_agg,traj_df],ignore_index=True)
    if verbose > 0:
        print("{} episodes, ({:.1f} min, DF {:.1f}MB) Avg reward {:.1f}+-{:.1f},".format(N_traj-n_traj_loaded,(time.time()-start)/60,sum(traj_df.memory_usage(deep=True))/2**20,np.mean(episode_rews),np.std(episode_rews)))

    if gif_path is not None:
        make_gif(img_array,gif_path)
    if path is not None:
        if os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        traj_df.to_pickle(path)
        if verbose>0:
            print('Saved {} trajs to {}'.format(N_traj,path))
    env.close()
    #np.random.seed()
    return traj_df

def make_gif(img_array,gif_path):
    #img_array = img_array[::4]
    fig = plt.figure(figsize=(img_array[0].shape[1] / 100.0, img_array[0].shape[0] / 100.0), dpi=img_array[0].shape[0]/2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    patch = ax.imshow(img_array[0])
    animate = lambda i: patch.set_data(img_array[i])
    gif = animation.FuncAnimation(plt.gcf(), animate, frames = len(img_array), interval=50)
    gif.save(gif_path, writer='imagemagick', fps=50)

##########    Build Network and Train

def nature_cnn_keras(inputs,seed=None):
    """ CNN architecture from the nature paper, input tensor (image) returns 1D tensor (features) """
    initializer = tf.glorot_uniform_initializer(seed=seed,)
    layer_1 = tf.keras.layers.Conv2D(filters=32, name='c1', kernel_size=8, strides=4, activation='relu',kernel_initializer=initializer)(inputs)
    layer_2 = tf.keras.layers.Conv2D(filters=64, name='c2', kernel_size=4, strides=2, activation='relu',kernel_initializer=initializer)(layer_1)
    layer_3 = tf.keras.layers.Conv2D(filters=64, name='c3', kernel_size=3, strides=1, activation='relu',kernel_initializer=initializer)(layer_2)
    layer_3 = conv_to_fc(layer_3)
    out = tf.keras.layers.Dense(units=512, name='fc1', activation='relu')(layer_3)
    return out

def keras_NN(Obs_shape,A_dim,H_dims=[8],linear=True,seed=None,n_components=1,cnn=False,NN_mid_as_feats=False,model_reg_coeff=None,clip_range=None):
    '''NN builder func
        Discrete out_dim is A_dim
    '''
    initializer = tf.glorot_uniform_initializer(seed=seed)
    reg = tf.keras.regularizers.l2(model_reg_coeff) if model_reg_coeff is not None else None
    if cnn:
        inp = tf.keras.Input(shape=Obs_shape)
        cast_inp = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32)/tf.cast(255.0,tf.float32))(inp)
        feats = nature_cnn_keras(cast_inp,seed=seed)
        feats2 = tf.keras.layers.Dense(units=32, name='fc2', activation='relu', kernel_initializer=initializer, bias_initializer=initializer,
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg)(feats)
        logits = tf.keras.layers.Dense(units=A_dim, name='pi', kernel_initializer=initializer, bias_initializer=initializer,
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg)(feats2)
        return tf.keras.Model(inputs=inp,outputs=logits)
    if linear:
        inp = tf.keras.Input(shape=Obs_shape)
        cast_inp = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32)/tf.cast(255.0,tf.float32))(inp)
        logits = tf.keras.layers.Dense(units=A_dim,input_shape=Obs_shape,kernel_initializer=initializer, bias_initializer=initializer,
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg,dtype=tf.float32)(cast_inp)
        return tf.keras.Model(inputs=inp,outputs=logits)
    else:
        inp = tf.keras.Input(shape=Obs_shape)
        cast_inp = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32)/tf.cast(255.0,tf.float32))(inp)
        middle = tf.keras.layers.Dense(H_dims[0],input_shape=Obs_shape,activation='relu',kernel_initializer=initializer, bias_initializer=initializer,
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg,dtype=tf.float32)(cast_inp)
        for i in range(1,len(H_dims)):
            middle = tf.keras.layers.Dense(H_dims[i],activation='relu',kernel_initializer=initializer, bias_initializer=initializer,
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg,dtype=tf.float32)(middle)
        logits = tf.keras.layers.Dense(A_dim,kernel_initializer=initializer,#activation='tanh',
                                        kernel_regularizer = reg, bias_regularizer = reg,activity_regularizer=reg,dtype=tf.float32)(middle)
        if clip_range is not None:
            print(clip_range)
            logits = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,clip_range[0],clip_range[1]))(logits)
        return tf.keras.Model(inputs=inp,outputs=logits)
def clone_model_and_weights(old_model):
    new_model = tf.keras.models.clone_model(old_model)
    new_model.set_weights(old_model.get_weights())
    return new_model

def train_model(model,dataframe,loss,learning_rate,N_epoch=20,batch_size=32,save_best_intermediate=False,model_prob_a=None,entropy_coeff=None,DISCRETE=True,
                steps_per_epoch=None,verbose=0,seed=None,delta=1e-6,adversary_f=None,df_test=None,test_loss=None,recompute_adversary=None,recompute_adversary_freq=1):
    '''trains keras model, either by taking N_epoch*steps_per_epoch optimization
       steps or until step size drops below delta'''
    is_fail = 'is_expert' in dataframe.columns
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataframe.loc[:,'loss'] = np.zeros((len(dataframe),))
    dataframe.loc[:,'adversary'] = np.ones((len(dataframe),))
    #dataframe.loc[:,'adversary_E'] = np.zeros((len(dataframe),))
    if DISCRETE:
        dataframe.loc[:,'model_prob_a'] = np.zeros((len(dataframe),))
        dataframe.loc[:,'model_a'] = np.zeros((len(dataframe),))
        model_a = lambda model_out,a: tf.gather_nd(model_out,a,batch_dims=1)
        dataframe.loc[:,'model_act_pre'] = np.argmax(batch_eval(model,np.vstack(dataframe['obs'].to_numpy())),axis=-1)
        dataframe.loc[:,'model_act_post']= -1*np.ones((len(dataframe),))

    train_results = dict()
    train_losses = []
    probs,advs = [],[]
    steps_per_epoch = steps_per_epoch or len(dataframe) #if None, take num_samp steps
    #steps_per_epoch = (steps_per_epoch//len(dataframe))*len(dataframe)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed); np.random.seed(seed)
    last_loss = 1e9
    print_freq = 10000
    start_time = time.time()
    best_weights = model.get_weights()
    best_ind = 0
    for epoch in range(N_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        random_inds = itertools.cycle(np.random.permutation(dataframe.index.values)) #random shuffle inds
        if is_fail:
            #print('All samps :',len(dataframe.index.values),'Learner only samps:',len(dataframe[dataframe['is_expert']==False].index.values))
            random_inds = itertools.cycle(np.random.permutation(dataframe[dataframe['is_expert']==False].index.values)) #random shuffle inds
        n_steps,print_count = 0,0
        while n_steps<steps_per_epoch:
            batch_indices = [next(random_inds) for i in range(min(batch_size,steps_per_epoch-n_steps))]
            with tf.GradientTape() as tape:
                loss_value = loss(model, dataframe.loc[batch_indices], adversary_f,model_prob_a,entropy_coeff=entropy_coeff)
                grads = tape.gradient(loss_value, model.trainable_variables)
            #Apply gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            last_loss = np.mean(loss_value)
            dataframe.loc[batch_indices,'loss'] = loss_value.numpy()
            epoch_loss_avg.update_state(loss_value)  # Add list of current losses
            n_steps += len(batch_indices)
            #Refit adversary after each gradient step (or after each epoch)
            if recompute_adversary_freq == 3:
                if recompute_adversary is not None:
                    adversary_f = recompute_adversary(model)
        if recompute_adversary_freq == 2:
            if recompute_adversary is not None:
                adversary_f = recompute_adversary(model)
        #FAIL takes the model with the lowest utility in the game
        train_losses.append(dataframe['loss'].mean())#epoch_loss_avg.result().numpy())
        epoch_loss_avg.reset_states()
        min_ind = np.argmin(train_losses)
        if train_losses[-1]<=train_losses[min_ind]:
            best_weights = model.get_weights()
            best_ind = min_ind
        
        if verbose>1:
            #if dataframe.isnull().values.any():
            #    print(dataframe.isnull().sum(axis=0))
            print('Loss',dataframe['loss'].mean(),'Loss recompute',np.mean(batch_avg_loss(loss,model,dataframe,adversary_f,model_prob_a,entropy_coeff)))
            tl_str = ' Test Loss: {:.3f}'.format(batch_avg_loss(test_loss,model,df_test)) if df_test is not None else ''
            print("Epoch {:03d}: Train Loss: {:.3f}{}".format(epoch+1,dataframe['loss'].mean(),tl_str))
        dataframe.loc[:,'loss'] = batch_loss(loss,model,dataframe,adversary_f,model_prob_a,entropy_coeff)
        #Debug printing
        if is_fail and verbose>1:

            N=20
            dataframe.loc[:,'adversary'] = adversary_f(np.vstack(dataframe['obs_next_ref'].to_numpy()),dataframe['t'].values)
            #dataframe.loc[:,'adversary_E'] = adversary_f(np.vstack(dataframe['obs_next_E'].to_numpy()),dataframe['t'].values)
            obs = np.vstack(dataframe['obs'].to_numpy())
            action_ref = tf.cast(np.vstack(dataframe['action_ref'].to_numpy()),tf.int32)
            dataframe.loc[:,'model_prob_a'] = batch_model_func_out_a(model_prob_a,model,obs,action_ref)
            dataframe.loc[:,'chosen'] = dataframe['model_prob_a']>.5
            dataframe.loc[:,'model_a'] = batch_model_func_out_a(model_a,model,obs,action_ref)
            
            
            smol = dataframe[dataframe['is_expert']==False].sort_values(['traj_ind','E_ind','i_agg'])
            print(smol[['action','action_ref','model_prob_a','model_a','adversary','loss','traj_ind','E_ind','i_agg']])
            advs.append(smol['adversary'].to_numpy())
            probs.append(smol['model_prob_a'].to_numpy())
            print('Chosen ({}): loss {:.3g}, f val {:.3g}, Non-Chosen ({}): loss {:.3g}, f val {:.3g}, All({}): loss {:.3g}, f val {:.3g}'.format(
                   dataframe[dataframe['chosen']].shape[0],
                   dataframe[dataframe['chosen']]['loss'].mean(),
                   dataframe[dataframe['chosen']]['adversary'].mean(),
                   dataframe[~dataframe['chosen']].shape[0],
                   dataframe[~dataframe['chosen']]['loss'].mean(),
                   dataframe[~dataframe['chosen']]['adversary'].mean(),
                   dataframe.shape[0],
                   dataframe['loss'].mean(),
                   dataframe['adversary'].mean()))
            print('Learner ({}): loss {:.3g}, f val {:.3g}, Expert ({}): loss {:.3g}, f val {:.3g}'.format(
                   dataframe[dataframe['is_expert']==False].shape[0],
                   dataframe[dataframe['is_expert']==False]['loss'].mean(),
                   dataframe[dataframe['is_expert']==False]['adversary'].mean(),
                   dataframe[dataframe['is_expert']==True].shape[0],
                   dataframe[dataframe['is_expert']==True]['loss'].mean(),
                   dataframe[dataframe['is_expert']==True]['adversary'].mean()))
            #print(dataframe['adversary'].min(),dataframe['adversary'].max(),dataframe['adversary'].mean(),dataframe['adversary'].std(),len(dataframe['adversary']))
            #print(dataframe[['obs','action','action_ref','action_ref_prob','adversary','adversary_E','model_prob_a','model_a','loss','E_ind']][(dataframe['model_prob_a']==1.0) & (dataframe['action_ref']==0) & ~dataframe['is_expert']].sort_values('E_ind'))
            #print(dataframe[['obs','action','action_ref','action_ref_prob','adversary','adversary_E','model_prob_a','model_a','loss','E_ind']][(dataframe['model_prob_a']==1.0) & (dataframe['action_ref']==1) & ~dataframe['is_expert']].sort_values('E_ind'))
            #print(dataframe[['obs','action','action_ref','action_ref_prob','adversary','adversary_E','model_prob_a','model_a','loss','E_ind']].sort_values('E_ind')[:N])

    #Done training, update with best weights
    if is_fail:
        dataframe.loc[:,'adversary'] = adversary_f(np.vstack(dataframe['obs_next_ref'].to_numpy()),dataframe['t'].values)
        model.set_weights(best_weights)
    
    train_results['epoch_losses'] = train_losses
    train_results['best_epoch_ind'] = best_ind
    if DISCRETE:
        dataframe.loc[:,'model_act_post'] = np.argmax(batch_eval(model,np.vstack(dataframe['obs'].to_numpy())),axis=-1)
        actions = tf.cast(np.vstack(dataframe['action_ref'].to_numpy()),tf.int32) if recompute_adversary is not None else tf.cast(np.vstack(dataframe['action'].to_numpy()),tf.int32)
        dataframe.loc[:,'model_a'] = batch_model_func_out_a(model_a,model,np.vstack(dataframe['obs'].to_numpy()),actions)
        dataframe.loc[:,'model_prob_a'] = batch_model_func_out_a(model_prob_a,model,np.vstack(dataframe['obs'].to_numpy()),actions)
        dataframe.loc[:,'chosen'] = dataframe['model_prob_a']>.5
        prob = dataframe[dataframe['is_expert']==False]['model_prob_a'] if ('is_expert' in dataframe.columns) else dataframe['model_prob_a']
        logit = dataframe[dataframe['is_expert']==False]['model_a'] if ('is_expert' in dataframe.columns) else dataframe['model_a']
        train_results['entropy'] = -(prob*prob.apply(np.log)).mean()
        train_results['entropy_coeff'] = entropy_coeff
        train_results['num_switched'] = len(dataframe[dataframe['model_act_pre']!=dataframe['model_act_post']])
        train_results['mean_abs_logit'] = logit.apply(np.abs).mean()
        train_results['min_abs_logit'] = logit.apply(np.abs).min()
    if is_fail and verbose>1:
        if 0:
            cs = ([c for c in plt.cm.get_cmap('tab20').colors]*4)[:len(probs[0])]
            plt.gca().set_prop_cycle(cycler('color', cs))
            plt.plot(probs,advs)
            plt.scatter(probs[-1],advs[-1],c=cs)
            plt.xlabel('Probability'); plt.ylabel('Utility')
        visualize_adversary(dataframe,False)
    if verbose>0:
        if is_fail:
            print('##   Utilities:',', '.join([f'{r:.3f}' for r in train_losses]),'Best',best_ind)
        if DISCRETE:
            print('##   Num switched {num_switched}, Entropy {entropy:.3f}, Mean abs logit {mean_abs_logit:.2f}, '\
                'Min abs logit {min_abs_logit:.3f}, Entropy Coeff {entropy_coeff:.1g}'.format(**train_results))
        tl_str = ' Test Loss: {:.3f}'.format(batch_avg_loss(test_loss,model,df_test)) if df_test is not None else ''
        tl_str += ' ({:.1f} min)'.format((time.time()-start_time)/60)
        print("Train Loss ({} Epochs): {:.3f}{}".format(epoch+1,epoch_loss_avg.result(),tl_str))
    #if 'is_expert' in dataframe.columns:
 #       dataframe[dataframe['is_expert']==False].hist(column='loss', bins=100)
  #  else:
   #     dataframe.hist(column='loss', bins=100)
    #plt.show()
    return train_results

def visualize_adversary(df,plot_chosen=False):
    X_all = np.vstack(df['obs'].to_numpy())
    pca = PCA(n_components=2)
    pca.fit(X_all)
    df_L = df[df['is_expert']==False].sort_values(['traj_ind','t'])
    ch_inds = (df_L['chosen']).to_numpy()
    N = sum(ch_inds)
    print('Chosen utility',(df_L[ch_inds]['adversary']*df_L[ch_inds]['model_prob_a'])[:N].mean(),
          'Other utility',(df_L[~ch_inds]['adversary']*df_L[~ch_inds]['model_prob_a'])[:N].mean())
    print('Chosen loss',df_L[ch_inds]['loss'][:N].mean(),
          'Other loss',df_L[~ch_inds]['loss'][:N].mean())
    plt.figure()
    plt.hist(df_L[ch_inds]['adversary'],100,label='chosen f',range=(df_L['adversary'].min(), df_L['adversary'].max()),alpha=.7)
    plt.hist(df_L[~ch_inds]['adversary'],100,label='not chosen f',range=(df_L['adversary'].min(), df_L['adversary'].max()),alpha=.7)
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(df_L[ch_inds]['loss'],100,label='chosen loss',range=(df_L['loss'].min(), df_L['loss'].max()),alpha=.7)
    plt.hist(df_L[~ch_inds]['loss'],100,label='not chosen loss',range=(df_L['loss'].min(), df_L['loss'].max()),alpha=.7)
    plt.legend()
    plt.show()
    E_pca = pca.transform(np.vstack(df['obs_next_ref'][df['is_expert']==True].to_numpy()))
    L_pca = pca.transform(np.vstack(df_L['obs_next_ref'].to_numpy()))
    #print(df_L[['obs','action_ref','obs_next_ref']][ch_inds][:N])
    #print(df_L[['obs','action_ref','obs_next_ref']][~ch_inds][:N])
    E_adv_f = df['adversary'][df['is_expert']==True].to_numpy()
    L_adv_f = df_L['adversary'].to_numpy()
    E_prob = df['model_prob_a'][df['is_expert']==True].to_numpy()
    L_prob = df_L['model_prob_a'].to_numpy()
    f_vals = [*E_adv_f,*L_adv_f]
    f_mean,f_std = np.mean(f_vals),np.std(f_vals)
    vmin,vmax = min(f_vals),max(f_vals)#f_mean-2*f_std,f_mean+2*f_std
    cms = ['Purples','Blues','Greens','Oranges','Reds']*10
    colors = ['tab:purple','tab:blue','tab:green','tab:orange','tab:red']*10
    
    if plot_chosen:
        plt.scatter(L_pca[ch_inds][:N,0],L_pca[ch_inds][:N,1],marker='+',
                    c=L_adv_f[ch_inds][:N],cmap='hsv',vmin=vmin,vmax=vmax)
        plt.scatter(L_pca[~ch_inds][:N,0],L_pca[~ch_inds][:N,1],marker='x',
                    c=L_adv_f[~ch_inds][:N],cmap='hsv',vmin=vmin,vmax=vmax)
        legend_elements = [Line2D([0], [0], marker='+', label='chosen'),Line2D([0], [0], marker='x', label='not chosen')]
    else:
        plt.scatter(E_pca[:,0],E_pca[:,1],marker='x',c=E_adv_f,cmap='Greys',vmin=vmin,vmax=vmax)
        #plt.colorbar()
        for i in pd.unique(df_L['i_agg']):
            i_inds = (df_L['i_agg']==i).to_numpy()
            plt.scatter(L_pca[i_inds][:,0],L_pca[i_inds][:,1],marker='o',
                        s=L_prob[i_inds]*100,c=L_adv_f[i_inds],cmap=cms[int(i)],vmin=vmin,vmax=vmax)
        legend_elements = [Line2D([0], [0], marker='x', color='w',markeredgecolor='tab:gray', label='Expert'),
                           *[Line2D([0], [0], marker='o', color='w',markerfacecolor=colors[int(i)], label=f'Learner {int(i)}') for i in pd.unique(df_L['i_agg'])]]
    plt.colorbar()
    plt.legend(handles = legend_elements)
    plt.show()
####### Loss functions

def softmax_cross_entropy(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    #Don't apply a softmax at the final layer!! This already does that for you!!
    a = np.hstack(dataset['action'].to_numpy())             # N x 1
    obs = np.vstack(dataset['obs'].to_numpy())              # N x O_dim
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)    # N x 1
    #print(a,obs,w)
    #print(tf.compat.v1.losses.sparse_softmax_cross_entropy(a,model(obs),weights=w))
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a,logits=model(obs))*w
    
    #tf.compat.v1.losses.sparse_softmax_cross_entropy(a,model(obs),weights=w) #This is the same as tf.nn... but with a tf.reduce_mean wrapped around
    #this is equivalent to -np.log(softmax(model(obs),axis=1))[:,a]
def zeroone_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    a = np.hstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    return tf.cast(tf.math.not_equal(a,tf.math.argmax(model(obs),axis=1)),tf.float32)*w
def L2_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    a = np.vstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    return tf.sqrt(tf.reduce_sum((model(obs)-a)**2,axis=1))*w
def mse_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    a = np.vstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    return tf.reduce_sum((model(obs)-a)**2,axis=1)*w
def L1_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    a = np.vstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    return tf.reduce_sum(tf.abs(model(obs)-a),axis=1)*w
def logcosh_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    a = np.vstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    return tf.reduce_sum(tf.log(tf.cosh(model(obs)-a)),axis=1)*w
def FAIL_loss_old_old(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):

    obs = np.vstack(dataset['obs'].to_numpy())
    w_E = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    is_expert = tf.cast(dataset['is_expert'].to_numpy(),tf.float32)
    a = tf.cast(np.vstack(dataset['action'].to_numpy()),tf.int32)
    adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next'].to_numpy()),dataframe['t'].values),tf.float32))
    action_weight_L = tf.gather_nd(model(obs),a,batch_dims=1)/tf.cast(np.hstack(dataset['action_ref_prob'].to_numpy()),tf.float32)
    #print(action_weight_L.shape,a.shape,model(obs).shape)
    FAIL = action_weight_L*adversary*(1-is_expert) + w_E*adversary*is_expert
    #print(FAIL,adversary,action_weight_L,w,is_expert,(1-is_expert))
    #raise NotImplementedError()
    return FAIL#tf.reduce_mean(FAIL)

def FAIL_loss_old(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    obs = np.vstack(dataset['obs'].to_numpy())
    w_E = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    #is_expert = tf.cast(dataset['is_expert'].to_numpy(),tf.float32)
    a = tf.cast(np.vstack(dataset['action_ref'].to_numpy()),tf.int32)
    adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next_ref'].to_numpy())),tf.float32))
    #adversary_E = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next_orig'].to_numpy())),tf.float32))
    action_weight_L = tf.gather_nd(model_prob_a(model(obs)),a,batch_dims=1)/tf.cast(np.hstack(dataset['action_ref_prob'].to_numpy()),tf.float32)
    #print(action_weight_L.shape,a.shape,model(obs).shape)
    #FAIL = (action_weight_L*adversary-adversary_E)*(1-is_expert) #+ w_E*adversary*is_expert
    FAIL = action_weight_L*adversary#*(1-is_expert) #+ w_E*adversary*is_expert
    #print(FAIL,adversary,action_weight_L,w,is_expert,(1-is_expert))
    #raise NotImplementedError()
    if 0:
        N=10
        print('obs',obs[:N])
        print('obs_next',np.vstack(dataset['obs_next'].to_numpy())[:N])
        print('obs_next_orig',np.vstack(dataset['obs_next_ref'].to_numpy())[:N])
        print('action',np.vstack(dataset['action'].to_numpy())[:N])
        print('action_orig',np.vstack(dataset['action_ref'].to_numpy())[:N])
        print('w_E',w_E[:N])
        print('is_expert',is_expert[:N])
        print('a',a[:N])
        print('adversary',adversary[:N])
        print('action_weight_L',action_weight_L[:N])
        raise Exception()
    return FAIL#tf.reduce_mean(FAIL)

def FAIL_loss(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    obs = np.vstack(dataset['obs'].to_numpy())
    a = tf.cast(np.vstack(dataset['action_ref'].to_numpy()),tf.int32)
    #adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next_ref'].to_numpy())),tf.float32))/tf.cast(np.hstack(dataset['action_ref_prob'].to_numpy()),tf.float32)
    adversary = tf.cast(dataset['adversary'].to_numpy()/dataset['action_ref_prob'].to_numpy(),tf.float32)
    #adversary_E = tf.cast(dataset['adversary_E'].to_numpy(),tf.float32)
    #pi_action_ref_given_obs = tf.gather_nd(model_prob_a(model(obs)),a,batch_dims=1)
    pi_action_ref_given_obs = model_prob_a(model(obs),a)
    entropy = -pi_action_ref_given_obs*tf.log(pi_action_ref_given_obs+1e-12)
    entropy_coeff = 0 if entropy_coeff is None else entropy_coeff
    loss = pi_action_ref_given_obs*adversary*w - entropy*entropy_coeff
    nanvals = np.isnan(loss)
    if nanvals.any():
        print(model(obs)[nanvals],pi_action_ref_given_obs[nanvals],adversary[nanvals],w[nanvals],entropy[nanvals])
        raise Exception()
    return pi_action_ref_given_obs*adversary*w - entropy*entropy_coeff #- adversary_E

def FAIL_loss_crossentropy(model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None):
    obs = np.vstack(dataset['obs'].to_numpy())
    a = tf.cast(np.hstack(dataset['action_ref'].to_numpy()),tf.int32)
    adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next_ref'].to_numpy())),tf.float32))/tf.cast(np.hstack(dataset['action_ref_prob'].to_numpy()),tf.float32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a,logits=model(obs))*adversary

def batch_loss_old(loss,model,dataset,adversary_f=None,model_prob_a=None,batch_size=2048):
    return sum([loss(model,dataset[i:i+batch_size],adversary_f,model_prob_a).numpy()*len(dataset[i:i+batch_size]) for i in range(0,len(dataset),batch_size)])/len(dataset)
def batch_avg_loss(loss,model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None,batch_size=2048):
    return np.mean(batch_loss(loss,model,dataset,adversary_f,model_prob_a,entropy_coeff,batch_size))
def batch_loss(loss,model,dataset,adversary_f=None,model_prob_a=None,entropy_coeff=None,batch_size=2048):
    return np.hstack([loss(model,dataset[i:i+batch_size],adversary_f,model_prob_a,entropy_coeff).numpy() for i in range(0,len(dataset),batch_size)])
def batch_eval(model,dataset,batch_size=2048):
    if len(dataset)<= batch_size:
        return model(dataset)
    return np.vstack([model(dataset[i:i+batch_size]).numpy() for i in range(0,len(dataset),batch_size)])
def batch_model_func_out_a(func_out_a,model,dataset,a,batch_size=2048):
    if len(dataset)<= batch_size:
        return func_out_a(model(dataset),a)
    return np.hstack([func_out_a(model(dataset[i:i+batch_size]),a[i:i+batch_size]).numpy() for i in range(0,len(dataset),batch_size)])
### Auxiliary

def sample_next_obs(env,state,action,state_prev=None,action_prev=None,n_avg=1):
    '''returns mean next obs over n_avg (sp~P(sp|s,a)) single step simulations'''
    obs_nexts = []
    for i in range(n_avg):
        obs = reset_env(env,state,state_prev,action_prev) #ATARI requires prev state/action for single step roll-in
        obs_nexts.append(env.step(action)[0])
    return np.mean(obs_nexts,axis=0)
def compute_action_prob(df,model,model_prob_a,batch_size=2048):
    p = np.zeros((len(df),))
    for b in range(0,len(df),batch_size):
        BS = min(len(df)-b,batch_size)
        #p[b:b+BS] = model_prob_a(model(np.vstack(df['obs'][b:b+BS].to_numpy())))[np.arange(BS),np.hstack(df['action'][b:b+BS].to_numpy())]
        p[b:b+BS] = model_prob_a(model(np.vstack(df['obs'][b:b+BS].to_numpy())),tf.cast(np.vstack(df['action_ref'][b:b+BS].to_numpy()),tf.int32))
    return p
def resample_next_states(df_L,sample_env,A_dim,n_samp=1,num_new=None,verbose=0,obs_postprocess = None):
    env_id = sample_env.unwrapped.spec.id
    obs_post = (lambda obs,**kw:obs) if obs_postprocess is None else obs_postprocess
    #print('resamp pre',df_L['i_agg'].max(),len(df_L))
    df_L = df_L.dropna(subset=['state_prev', 'state','action_prev','action'])
    num_new = num_new or len(df_L)
    #print('resamp pre',df_L['i_agg'].max(),len(df_L))
    n_samp = min(n_samp,A_dim)
    df = pd.concat([df_L[-num_new:] for n in range(n_samp)],ignore_index=True)
    #print('num_new',num_new,len(df))
    #print(df['action'][:10])
    #df['action_orig'] = df['action']
    #df['obs_next_orig'] = df['obs_next']
    if n_samp==A_dim:   #Try every action
        df['action_ref'] = np.array([i for i in range(A_dim) for j in range(len(df)//A_dim)],dtype=np.int32)
    else:               #Randomly subsample actions
        df['action_ref'] = np.random.randint(A_dim,size=(len(df),1))
    #print(df['action'][:10])
    #df['action'].plot.hist(bins=6)
    #plt.show()
    df['action_ref_prob'] = np.ones((len(df),1))/A_dim             #Calculate action probability
    start = time.time()
    #Should this be action_prev=row['action_ref']? (previously it was row['action'])
    df['obs_next_ref'] = df.apply(lambda row: obs_post(sample_next_obs(sample_env,row['state'],row['action_ref'],row['state_prev'],row['action_prev']),action_prev=row['action_ref'],env_id=env_id),axis=1)
    df['sp_dist'] = df.apply(lambda row: np.linalg.norm(row['obs_next']-row['obs_next_ref']),axis=1)
    #df['sp_dist'].plot.hist(bins=100)
    plt.show()
    N=10
    #print('obs_next\n',np.vstack(df['obs_next'].to_numpy())[:N])
    #print('obs_next_orig\n',np.vstack(df['obs_next_orig'].to_numpy())[:N])
    if verbose>0:
        print('Done forward simulating ({:.1f})'.format((time.time()-start)/60))
    df_out = pd.concat([df_L[:-num_new],df],ignore_index=True)
    df_out = df_out.dropna().reset_index(drop=True)
    #print('resamp post',df_out['i_agg'].max())
    return df_out
    
def FORWARD_sample_next_states(df_L,sample_env,A_dim,n_samp=1,verbose=0,obs_postprocess=None):
    # We are just using s' from df_L
    env_id = sample_env.unwrapped.spec.id
    obs_post = (lambda obs,**kw:obs) if obs_postprocess is None else obs_postprocess
    #df_L = df_L.dropna(subset=['state_next', 'state','action'])
    df_out = pd.DataFrame()
    # s of df_out = s' of df_L
    df_out[['obs','state','state_prev','action_prev']] = df_L[['obs_next','state_next','state','action']]
    df_out = pd.concat([df_out for n in range(n_samp)],ignore_index=True)
    
    # Generate reference actions
    if n_samp==A_dim:   #Try every action
        df_out['action_ref'] = np.array([i for i in range(A_dim) for j in range(len(df_out)//A_dim)],dtype=np.int32)
    else:               #Randomly subsample actions
        df_out['action_ref'] = np.random.randint(A_dim,size=(len(df_out),1))
    df_out['action_ref_prob'] = np.ones((len(df_out),1))/A_dim             #Calculate action probability
    df_out['weight'] = np.ones((len(df_out),1))
    #df_out['action'] = df_out['action_ref']
    # Resample
    start = time.time()
    df_out['obs_next_ref'] = df_out.apply(lambda row: obs_post(sample_next_obs(sample_env,row['state'],row['action_ref'],
                                                                               row['state_prev'],row['action_prev']),
                                                               action_prev=row['action_ref'],env_id=env_id),axis=1)
    if verbose>0:
        print('Done forward simulating ({:.1f})'.format((time.time()-start)/60))
    N=10
    #print('obs_next\n',np.vstack(df['obs_next'].to_numpy())[:N])
    #print('obs_next_orig\n',np.vstack(df['obs_next_orig'].to_numpy())[:N])
    return df_out

def setup_training_dataframe(alg,df_E,df_L=None,pi_E=None,num_new=None):
    if alg == 'DaD' and df_L is not None:
        df_train = pd.merge(df_L[['obs','t','E_ind']],df_E[['action','weight']].loc[df_L['E_ind']].reset_index(drop=True),left_index=True,right_index=True)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:
        df_E.loc[:,'obs_next_ref'] = df_E['obs_next']
        #df_E['obs_next_E'] = df_E['obs_next']
        df_E.loc[:,'action_ref'] = df_E['action']
        #df_L = pd.merge(df_L,df_E[['obs_next_E']].loc[df_L['E_ind']].reset_index(drop=True),left_index=True,right_index=True)
        df_train = pd.concat([df_L,df_E],ignore_index=True).fillna(1)
        df_train['is_expert'] = np.hstack([np.zeros((len(df_L),)),np.ones((len(df_E),))])
        #df_train['loss'] = np.zeros((len(df_train),))
    elif alg in ['DAgger'] and pi_E is not None:
        df_train = df_L.copy()
        df_train['action'] = df_train.apply(lambda row : pi_E(row['obs']),axis=1)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-Cov','BC']: #BC, ALICE-Cov
        df_train = df_E.copy()
    else:
        raise NotImplementedError('Alg {} training dataframe setup not implemented'.format(alg))
    return df_train

def estimate_ratio_on_samps(x_L,x_E,feature_map_pipeline='linear',warm_start=False,verbose=0):
    '''Given distributions x_L,x_E (lists of items), we flatten them, pass them
     through the feature_map_pipeline specified, and perform logistic regression,
     then return a stacked array [x_L,x_E] of ratio estimates r(x) = P_L(x)/P_E(x)

     feature_map_pipeline applies the following mappings space-delineated in order
        pca-[n]         Principal Component analysis with n components
        poly-[d]        Polynomial feature map of degree d
        std_scaler      StandardScaler, shifts dataset to zero mean, unit variance
        qtl_transform   QuantileTransform, squishes dataset to 0-1 range by quantile
        rff-[n]         RandomFourierFeatures with n components

     ie. feature_map_pipeline = 'std_scaler pca-100 poly-2'   '''
    
    X = np.stack([x.flatten() for x in itertools.chain(x_L,x_E)])
    N_L,N_E = len(x_L),len(x_E)
    y = np.hstack([np.zeros((N_L,)),np.ones((N_E,))])
    pipeline_list = str_to_fm_pipeline(feature_map_pipeline)
    if pipeline_list[-1][0]=='RuLSIF':
        feature_map = Pipeline(pipeline_list)
        X_featurized = feature_map.fit_transform(X)
        #result = densratio(X_featurized[N_L:],X_featurized[:N_L],sigma_range="auto",lambda_range="auto",verbose=False)
        #result = densratio(X_featurized[N_L:],X_featurized[:N_L],sigma_range=[.0001,.001,.01,.1,1],lambda_range=[.0001,.001,.01,.1,1],verbose=False)
        result = densratio(X_featurized[N_L:],X_featurized[:N_L],sigma_range=[.01,.1,1],lambda_range=[.0001,.001,.01,.1],verbose=False)
        ratio = result.compute_density_ratio(X_featurized)    
        print('JS lam,sig:',result.lambda_,result.kernel_info.sigma)
    else:
        pipeline_list.append(('lr',LogisticRegression(warm_start=warm_start)))
        featurized_LR = Pipeline(pipeline_list)
        featurized_LR.fit(X,y)
        proba = featurized_LR.predict_proba(X) #class 1 is learner, class 0 is expert
        ratio = proba[:,0]*N_E/(proba[:,1]*N_L+EPS)
    if verbose>3:
        if 0:
            print('Generating TSNE, get some popcorn')
            X_tsne = TSNE(n_iter=250,perplexity=10).fit_transform(X)
        if 1:
            print('Generating PCA')
            X_tsne = PCA(n_components=2).fit_transform(X)      
        plt.scatter(X_tsne[N_L:,0],X_tsne[N_L:,1],label='E',alpha=.7)
        plt.scatter(X_tsne[:N_L,0],X_tsne[:N_L,1],label='L',marker='+',alpha=.7)
        plt.legend()
        plt.show()
    return ratio
def str_to_fm_pipeline(feature_map_pipeline='linear'):
    pipeline_list = []
    for ind,fm in enumerate(feature_map_pipeline.split(' ')):
        name,num = fm.split('-')[0],int(fm.split('-')[1]) if '-' in fm else None
        if name == 'linear':
            pipeline_list.append((f'{ind}-{name}', None))
        elif name =='RuLSIF':
            pipeline_list.append((f'{name}', None))
            break
        elif name =='poly':
            deg = num or 2
            pipeline_list.append((f'{ind}-{name}-{deg}', PolynomialFeatures(degree=deg)))
        elif name =='meanscaler':
            pipeline_list.append((f'{ind}-{name}', StandardScaler(with_std=False)))
        elif name =='stdscaler':
            pipeline_list.append((f'{ind}-{name}', StandardScaler(with_mean=False)))
        elif name =='standardscaler':
            pipeline_list.append((f'{ind}-{name}', StandardScaler()))
        elif name =='quantilescaler':
            pipeline_list.append((f'{ind}-{name}', QuantileTransformer()))
        elif name =='pca':
            n_comp = num or 100
            pipeline_list.append((f'{ind}-{name}-{n_comp}', PCA(n_components=n_comp)))
        elif name =='rff':
            n_comp = num or 100
            pipeline_list.append((f'{ind}-{name}-{n_comp}', RBFSampler(n_components=n_comp)))
        elif name =='nys':
            n_comp = num or 100
            pipeline_list.append((f'{ind}-{name}-{n_comp}', Nystroem(n_components=n_comp)))
        else:
            pipeline_list.append((f'{ind}-{name}', None))
        #if name == 'signsplit':
        #    pipeline_list.append((name+str(ind), FunctionTransformer(lambda X: np.hstack([X*(X>0),-X*(X<0)]))))
    return pipeline_list

def js_from_samples(X1, X2):
    #X1, X2: n x d matrices
    X1 = np.stack([x.flatten() for x in X1])
    X2 = np.stack([x.flatten() for x in X2])
    mean_0 = np.mean(X1, axis = 0)
    std_0 = np.std(X1, axis = 0) + 1e-7
    mean_1 = np.mean(X2,axis=0)
    std_1 = np.std(X2,axis=0) + 1e-7

    kl = 0.5*(np.sum(std_0/std_1) + (mean_0-mean_1).dot(np.diag(1./std_1)).dot(mean_0-mean_1)
              - X1.shape[1] + np.log(np.sum(std_1)/np.sum(std_0)))

    ikl = 0.5*(np.sum(std_1/std_0) + (mean_1 - mean_0).dot(np.diag(1./std_0)).dot(mean_1 - mean_0) 
             - X1.shape[1] + np.log(np.sum(std_0)/np.sum(std_1)))

    return (kl+ikl)/2.
def fit_adversary(x_0,x_1,w_0=None,w_1=None,feature_map_pipeline='linear',NN_mid_as_feats=False,model=None,Obs_shape=None):
    X = np.stack([x.flatten() for x in itertools.chain(x_0,x_1)])
    N_0,N_1 = len(x_0),len(x_1)
    w_0,w_1 = np.ones((N_0)) if w_0 is None else w_0, np.ones((N_1)) if w_1 is None else w_1
    c = np.hstack([w_0/N_0,-w_1/N_1])
    if NN_mid_as_feats:
        inter_output_model = tf.keras.Model(model.input, model.get_layer(index = -2).output)
        def NN_featurize(X):
            f_X = batch_eval(inter_output_model,np.reshape(X,[-1,*Obs_shape])) #inter_output_model(np.reshape(X,[-1,*Obs_shape]))
            #print(f_X[0].numpy()[:10])
            return f_X
        #featurize = lambda X :inter_output_model(np.reshape(X,[-1,*Obs_shape]))
        #X_featurized = featurize(X)
    else:
        def NN_featurize(X):
            X_np = np.stack([x.flatten() for x in X])
            #return np.hstack([X_np*(X_np>0),-X_np*(X_np<0)])   #If you only want positive features, do this.
            return X_np
        #NN_featurize = lambda X:np.stack([x.flatten() for x in X])

    pipeline_list = str_to_fm_pipeline(feature_map_pipeline)
    feature_map = Pipeline(pipeline_list)
    X_featurized = feature_map.fit_transform(NN_featurize(X))
    featurize = lambda X: feature_map.transform(X)

    mu = np.dot(c,X_featurized)
    w = mu/np.linalg.norm(mu)
    #print('mu',mu)
    #print('w',w)
    #w = np.ones_like(w)
    adversary_f = lambda X,*args: np.inner(w,featurize(NN_featurize(X)))
    #print(w)
    #adversary_f = lambda X: np.mean(X,axis=1)
    return adversary_f#,w,NN_featurize

def fit_adversary_t(x_0,x_1,w_0=None,w_1=None,feature_map_pipeline='linear',NN_mid_as_feats=False,model=None,Obs_shape=None,t_0=None,t_1=None,H=1):
    if H is None:
        return fit_adversary(x_0,x_1,w_0,w_1,feature_map_pipeline,NN_mid_as_feats,model,Obs_shape)
    t_vals = np.unique([*t_0,*t_1])
    # assert t_vals==np.arange(t_vals.min(),t_vals.max()+1) #Assume ts are not scattered randomly but instead one clean range
    min_max = min(max(t_0),max(t_1))
    t_buckets = [[i for i in range(t,min(t+H,min_max+1))] for t in range(t_vals.min(),min_max+1,H)]
    t_buckets[-1].extend([*range(min_max+1,t_vals.max()+1)]) #stick all excess inds in last bucket
    adversaries = []
    bucket_map = dict()
    t0_inds = [[ind for ind, val in enumerate(t_0) if val in bucket] for bucket in t_buckets]
    t1_inds = [[ind for ind, val in enumerate(t_1) if val in bucket] for bucket in t_buckets]
    for ind,bucket in enumerate(t_buckets):
        ind0,ind1 = t0_inds[ind],t1_inds[ind]
        w0 = w_0[ind0] if w_0 is not None else None
        w1 = w_1[ind1] if w_1 is not None else None
        adversaries.append(fit_adversary(x_0[ind0],x_1[ind1],w0,w1,feature_map_pipeline,NN_mid_as_feats,model,Obs_shape))
        bucket_map.update({t:ind for t in bucket})
    adversary_t = lambda X,t,*args : np.array([adversaries[bucket_map[t[i]]](X[i:i+1]) for i in range(len(t))])
    return adversary_t
    
def verify_adversary(N=1000,d=5,feature_map_pipeline='linear'):
    ''' Running this requires you modify fit_adversary to output adversary_f,w,NN_featurize'''
    N_samp0,N_samp1 = N//2,N//2
    dim =d
    mu0 = np.ones((dim,))*5
    mu1 = mu0+np.ones((dim,))*(dim**2)
    mu2 = copy.copy(mu1)
    mu2[1] = mu0[1]-(dim**2)
    sig0 = np.ones((dim,))*2*dim**.25
    sig1 = np.ones((dim,))*dim**.25
    sig2 = np.ones((dim,))*dim**.25
    eps = 1e-9
    p_0 = lambda x : np.exp(np.sum(-.5*((mu0-x)/sig0)**2-np.log(sig0)-np.log(2*np.pi)/2,axis=1))
    p_1 = lambda x : np.exp(np.sum(-.5*((mu1-x)/sig1)**2-np.log(sig1)-np.log(2*np.pi)/2,axis=1))
    r_true = lambda x : p_1(x)/(p_0(x)+EPS)+EPS

    x0 = np.random.randn(N_samp0,dim)*sig0 + mu0
    x1 = np.random.randn(N_samp1,dim)*sig1 + mu1
    x2 = np.random.randn(N_samp1,dim)*sig2 + mu2
    #y = np.hstack([np.zeros((N_samp0,)),np.ones((N_samp1,))])
    #X = np.vstack([x0,x1])
    adversary_f = fit_adversary(x0,x1,feature_map_pipeline=feature_map_pipeline)
    print('Mean diff between train distributions', np.mean(adversary_f(x0)-adversary_f(x1)))
    print('Mean diff against orthogonal distribution',np.mean(adversary_f(x0)-adversary_f(x2)))
    f_vals = [*adversary_f(x0),*adversary_f(x1)]
    cm = 'hsv'
    vmin,vmax = min(f_vals),300#max(f_vals)
    plt.scatter(x0[:,0],x0[:,1],marker='x',label='X0',c=adversary_f(x0),cmap=cm,vmin=vmin,vmax=vmax)
    plt.scatter(x1[:,0],x1[:,1],marker='o',label='X1',c=adversary_f(x1),cmap=cm,vmin=vmin,vmax=vmax)
    plt.scatter(x2[:,0],x2[:,1],marker='+',label='X2 (diff dist)',c=adversary_f(x2),cmap=cm,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.legend()
    plt.show()

def save_pkl_to_df(path):
    df_columns = ['obs','obs_next','state','state_next','state_prev','action','action_prev','rew','t','traj_ind','weight_action','weight','E_ind']
    trajs = pickle.load(open(path,'rb'))
    samp_list = []
    for traj_ind,traj in enumerate(trajs):
        for t,s in enumerate(traj):
            env_state_prev = None if t==0 else samp_list[-1][2]
            action_prev = None if t==0 else samp_list[-1][5]
            samp_list.append([s['o'],s['op'],s['s'],s['sp'],env_state_prev,s['a'],action_prev,s['r'],t,traj_ind,1.0,1.0,len(samp_list)])
    traj_df = pd.DataFrame(samp_list,columns=df_columns)
    traj_df.to_pickle(path+'.xz')
    
def save_df_to_pkl(df,path):
    keymap = {'obs':'o','action':'a','rew':'r','obs_next':'op','t':'t','state':'s','state_next':'sp'}
    traj_inds = df['traj_ind'].drop_duplicates().values
    trajs = []
    for ind in traj_inds:
        traj = []
        for rownum,s in df[df['traj_ind']==ind].iterrows():
            traj.append({keymap[k]:s[k] for k in keymap})
        trajs.append(traj)
    pickle.dump(trajs,open(path,'wb'))

def load_agg_save_safe(path,results_list=[]):
    '''loads df (if exists), appends df in results_list, saves and returns combined df'''
    results_list = [results_list] if type(results_list) is not list else results_list #idiot proof
    if len(results_list)>0:
        if os.path.exists(path):
            cols = pd.read_csv(path,nrows=1).columns
        else:
            cols = results_list[0].columns
            os.makedirs(os.path.dirname(path),exist_ok=True)
        if len(cols.difference(results_list[0].columns))==0: #Equality up to permutation
            with portalocker.Lock(path) as f:
                results_df = pd.concat(results_list,ignore_index=True)[cols]
                results_df.to_csv(f,index=False,header=f.tell()==0) #Adds header only to first line
        else:
            new_df = pd.concat(results_list,ignore_index=True)[cols]
            old_df = pd.read_csv(path)
            joint_df = pd.concat([old_df,new_df],ignore_index=True)
            joint_df.to_csv(path)
    return pd.read_csv(path)
    
def load_agg_save(path,results_list=[]):
    return load_agg_save_safe(path,results_list)

def plot_results(df,xaxis,yaxis,lines='alg',filters=None,**plotattrs):
    '''
    Averages accross all columns not specified in constants
    lines - string, list, or dataframe to select which lines to plot
        string or list chooses column(s) from dataframe and plots all unique
            entries/combinations as a separate line.
        dataframe plots each row from dataframe as a separate line
    filters - dict where key is attribute and value is list of permissible
    
    '''
    if type(df) is str:
        df = load_agg_save_safe(df)
    #Set Default Plot Attributes
    CI_style = plotattrs.setdefault('CI_style','polygon') #Confidence interval style 'polygon' or 'errorbar'
    colormap = plotattrs.setdefault('colormap','tab10')
    linestyles = plotattrs.setdefault('linestyles',['-'])
    xscale = plotattrs.setdefault('xscale','linear') #'log' or 'linear'
    yscale = plotattrs.setdefault('yscale','linear') #'log' or 'linear'
    linewidth = plotattrs.setdefault('linewidth',3)
    incl_exp = plotattrs.setdefault('incl_exp',False)
    incl_rand = plotattrs.setdefault('incl_rand',False)
    legend_on = plotattrs.setdefault('legend_on',True)
    ylabel_on = plotattrs.setdefault('ylabel_on',True)
    save_dir = plotattrs.setdefault('save_dir',None)
    env_id = plotattrs.setdefault('env_id',df['env_id'][0])
    exp_name = plotattrs.setdefault('exp_name','')
    legend = plotattrs.setdefault('legend',None)

    #Make sure filter values are lists, even if they only have one item
    filters = {k:([filters[k]] if type(filters[k]) is not list else filters[k]) for k in filters} if filters is not None else None

    # Gather lines and filter for desired attrs
    lines = [lines] if type(lines) is str else lines
    lines_df = df[lines].drop_duplicates().dropna() if type(lines) is list else lines
    if 'alg' in lines:
        if not incl_exp:
            lines_df = lines_df[~(lines_df['alg'].isin(['Expert']))]
        if not incl_rand:
            lines_df = lines_df[~(lines_df['alg'].isin(['Random']))]
    lines_df = lines_df.sort_values(lines)
    if filters is not None:
        df = df[(df[filters.keys()].isin(filters)).all(axis=1) | (df['alg'].isin(['Expert']) & incl_exp) | (df['alg'].isin(['Random']) & incl_rand)]

    #Set colormap, linestyles,axis labels, legend label formats, title
    if colormap in DISCRETE_COLORMAPS:
        cs = [c for c in plt.cm.get_cmap(colormap).colors]*2 #Discrete colormaps
    else: #Contiuous colormaps
        cs = [plt.cm.get_cmap(colormap)(val) for val in np.linspace(1,0,len(lines_df))] 
    lss = [ls for i in range(len(lines_df)) for ls in linestyles]
    xlabel = {'N_E_traj':'Number of Expert Trajectories',
              'N_E_samp':'Number of Expert Samples'}.get(xaxis,xaxis)
    ylabel = {'reward':'On-Policy Reward',
              'loss_train':'Training Loss',
              'loss_test':'Validation Loss',
              'w_ESS':'Effective Sample Size'}.get(yaxis,yaxis)
    leg_format = lambda col: {'total_opt_steps':'{:.2g}'}.get(col,'{}')
    leg_label = lambda col: {'total_opt_steps':'# opt','entropy_coeff':'w_H','drop_first':'drop',
                             'model_reg_coeff':'reg','learning_rate':'lr','horizon_weight_offset_exp':'hwoe',
                             'recent_samp_priority_exp':'pe','adversary_t_bucket_size':'bs','pair_with_E':'pair',
                             'n_feats':'nf'}.get(col,col)

    title = plotattrs.setdefault('title',' '.join([env_id,exp_name,ylabel]))

    #Add lines
    figname = '-'.join([env_id,exp_name,xaxis,yaxis])
    plt.figure(figname)
    lines_df = pd.DataFrame(lines_df).reset_index(drop=True) #Handle single line
    for i,line in lines_df.iterrows():
        label = ', '.join([leg_label(k)+' = '+leg_format(k).format(v) if k!='alg' else v for k,v in line.items()])
        if 'alg' in lines:
            label = line['alg'] if line['alg'] in ['Expert','Random'] else label
        label = legend[i] if legend is not None else label
        df_line = df[(df[line.keys()]==line.values).all(axis=1)].sort_values(xaxis)
        x = df_line[xaxis].drop_duplicates().to_numpy()
        n = np.array([len(df_line[df_line[xaxis]==xi][yaxis]) for xi in x])
        y = np.array([df_line[df_line[xaxis]==xi][yaxis].mean() for xi in x])
        yerr = np.array([df_line[df_line[xaxis]==xi][yaxis].std() for xi in x])
        if (yaxis=='reward') and (n==1).all() and ('reward_std' in df_line.columns):
            yerr = np.squeeze(np.array([df_line[df_line[xaxis]==xi]['reward_std'].to_numpy() for xi in x]))

        if CI_style == 'errorbar':
            plt.errorbar(x,y,yerr=yerr,c=cs[i],ls=lss[i],label=label,lw=linewidth)
        if CI_style == 'polygon':
            plt.plot(x,y,c=cs[i],ls=lss[i],label=label,lw=linewidth)
            xy = np.hstack((np.vstack((x,y + yerr)),np.fliplr(np.vstack((x,y - yerr))))).T
            plt.gca().add_patch(Polygon(xy=xy, closed=True, fc=cs[i], alpha=.2))
        

    plt.xscale(xscale)
    plt.yscale(yscale)
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend_on:
        plt.legend(loc='upper left')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir,figname))

def print_results(env_ids,results_dir,exp_name,algs=['BC','BC+H'],params = None, filters = None,final=True,format='print',data='reward'):
    # Format filters so all values are lists
    filters = {k:([filters[k]] if type(filters[k]) is not list else filters[k]) for k in filters} if filters is not None else dict()
    params = [params] if type(params) is str else params if params is not None else []
    cols = list(set(params) | set(filters.keys()))
    
    # Build header string and calculate width
    N_cols = len(cols)
    N_alg = len(algs)
    alg_w = 18
    alg_hdrs = [(alg + ' (# avg)')[:18] for alg in algs] 
    if format=='latex':
        hdr_str = (' {:<20} &' + ' {:^15} &'*N_cols +' {:^14} &'+' {:^18} &'*N_alg).format('Environment',*[k[:15] for k in cols],'Expert (rew)',*alg_hdrs)
    elif format=='csv':
        hdr_str = ('{},' + ' {},'*N_cols +' {},'+' {},'*N_alg).format('Environment',*[k[:15] for k in cols],'Expert (rew)',*alg_hdrs)
    else:
        hdr_str = ('|{:^20}|' + '{:^10}|'*N_cols +'{:^12}|'+'{:^18}|'*N_alg).format('Environment',*[k[:10] for k in cols],'Expert (rew)',*alg_hdrs)
        format = 'print'
    w = len(hdr_str)
    
    datalist = [data] if type(data) is str else data
    for data in datalist:
        if format=='print':
            print('-'*w+'\n|{0:^{1}}|\n'.format(data,w-2)+'-'*w)
            print(hdr_str)
            print('-'*w)
        else:
            print(data)
            print(hdr_str)
        for env_id in env_ids:
            if env_id[-2]=='_':
                df = load_agg_save(results_dir+'results-'+env_id[:-2]+'--'+exp_name+'.csv')
            else:
                df = load_agg_save(results_dir+'results-'+env_id+'--'+exp_name+'.csv')
            if final:
                df = df[df['final']==True]
            train_r_mean,train_r_std = df.loc[df.index[0],['train_r_mean','train_r_std']]
            
            lines_df = df[cols].drop_duplicates().dropna()
            lines_df = lines_df.sort_values(cols)
            if len(filters)>0:
                print('Filtered')
                df = df[(df[filters.keys()].isin(filters)).all(axis=1)]
            for i,line in lines_df.iterrows():
                line_str = {'print':'|{:^20}|'+'{:^10}|'*N_cols + '{:>6.0f}+-{:<4.0f}|',
                            'latex':' {:<20} '+'& {:^15} '*N_cols + '& ${:.1f}\pm {:.1f}$ ',
                            'csv':'{}'+', {}'*N_cols + ', {:.1f}+-{:.1f} ',}[format].format(
                                env_id,*[str(v) for v in line.values],train_r_mean,train_r_std)
                for alg in algs:
                    linedict = {'alg':alg,**{k:v for k,v in zip(line.keys(),line.values)}}
                    index = (df[linedict.keys()]==linedict.values()).all(axis=1)
                    if sum(index)==0:
                        #print((df[linedict.keys()]==linedict.values()).sum())
                        #print(linedict.keys(),linedict.values())
                        pass
                    df_line = df[(df[linedict.keys()]==linedict.values()).all(axis=1)]
                    data_vec = df_line[data].to_numpy()
                    #data_vec = np.array([np.mean(df[index & (df['opt_seed']==i)][data].to_numpy()) for i in pd.unique(df[index]['opt_seed'])])
                    data_mean = data_vec.mean() if len(data_vec)>0 else np.nan
                    if data == 'reward':
                        N_pts = df_line['N_test_rollout'].to_numpy()
                        rew_stds = df_line['reward_std'].to_numpy()
                        pooled_std = (np.sum(N_pts*(rew_stds**2+data_vec**2))/N_pts.sum()-data_mean**2)**.5 if len(data_vec)>0 else np.nan
                        # If all means are equal and N are the same, then pooled std is just mean of stds :O
                        data_str = {'print':'{:>7.0f}+-{:<5.0f} ({})|','csv':',{:.1f}+-{:.1f} ({})',
                                    'latex':'& ${:.1f}\pm {:.1f} ({})$ '}[format].format(data_mean,pooled_std,len(data_vec))
                    else:
                        data_str = {'print':'{:^14.5g} ({})|','latex':'& ${:.5f} ({})$ ','csv':',{:.5f} ({})'}[format].format(data_mean,len(data_vec))
                    line_str += data_str
                print(line_str)
        if format=='print':
            print('-'*w)

def argsparser():
    parser = argparse.ArgumentParser("Implementations of ALICE alg and IL baselines")
    parser.add_argument('alg', help='IL algorithm to run', default='BC', choices=['BC','DaD','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL','Expert','Random','DAgger'])
    parser.add_argument('--env_id', help='environment ID', default='CartPole-v1',required=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='CartPole-v1expert_traj_linear_deterministic.p')
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    parser.add_argument('--num_hid_layers', help='number of hidden layer in policy: default zero, i.e., linear policy', type=int, default=2)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_timesteps', help='total number of samples', type=int, default=1e6)
    parser.add_argument('--min_max_game_iteration', help='number of iterations in each min-max game', type = int, default = 500)
    parser.add_argument('--num_roll_in', help="num of roll ins in online train model", type=int,default=100)
    parser.add_argument('--adam_lr', help='learning rate in adam', type=float, default=1e-2)
    parser.add_argument('--l2_lambda',help='l2 regularization lambda', type=float, default=1e-7)
    parser.add_argument('--horizon', help='horizon of episode, num of policies should be horizon - 1', type=int, default = 100)
    parser.add_argument('--num_expert_trajs', help='number of expert trajectories', type=int, default=500)
    parser.add_argument('--warm_start', help='initlize pi with the previous trained one', type=int, default=0)
    parser.add_argument('--mixing',help='mixing the previous policy with uniform policy for pi_ref', type=int,default=0)

    return parser.parse_args()

def alg_runner(alg,env_id,**kwargs):
    '''
    Runs one of the following algorithms in environment [env_id] with the keyword
    args as established below.
    
    Algorithm options:
        BC              Behavioral Cloning
        ALICE-Cov       ALICE with density ratio correction
        ALICE-FAIL      ALICE with next state lookahead moment matching
        ALICE-Cov-FAIL  ALICE with both ^^
        DAgger          Dataset Aggregation for online expert querying
        Expert          Passthrough to provide expert performance results for benchmarking purposes
        Random          Passthrough to provide random performance results for benchmarking purposes
    '''
    passed_kwargs = copy.copy(kwargs)
    
    #useful booleans
    is_atari = env_id in ATARI_ENVS
    is_cov =  alg in ['ALICE-Cov','ALICE-Cov-FAIL']
    is_fail = alg in ['ALICE-FAIL','ALICE-Cov-FAIL']
    is_alice = is_cov or is_fail
    is_mujoco = env_id in MUJOCO_ENVS
    
    #########################################################################################
    ################################### Keyword Arguments ###################################
    #########################################################################################
    
    ### Play with these
    FORWARD = kwargs.setdefault('FORWARD',False)                                # FORWARD execution: one policy per timestep. No aggregation
    N_agg_iter = kwargs.setdefault('N_agg_iter',10 if alg not in ['Expert','Random'] else 1)
    N_E_traj = kwargs.setdefault('N_E_traj',10)                                 # Number of expert trajectories to use as test data
    N_ALICE_traj = kwargs.setdefault('N_ALICE_traj',N_E_traj)                   # Number of learner trajectories to generate (generates N_agg
    N_epoch = kwargs.setdefault('N_epoch',5) #Number of training epochs
    opt_steps_per_iter = kwargs.setdefault('total_opt_steps',500000)//N_agg_iter
    add_history = kwargs.setdefault('add_history',False)
    kill_feats = kwargs.setdefault('kill_feats',None)
    verbose = kwargs.setdefault('verbose',0)
    run_seed = kwargs.setdefault('run_seed',None)
    opt_seed = kwargs.setdefault('opt_seed',None) #If you are initializing from scratch, this is the weight initializer seed, otherwise this is just batch shuffle seed
    density_ratio_feature_map = kwargs.setdefault('density_ratio_feature_map','linear' if is_cov else None)
    adversary_feature_map = kwargs.setdefault('adversary_feature_map','poly 2' if is_fail else None)
    ### Probably don't play with these
    learning_rate = kwargs.setdefault('learning_rate',0.01)
    learning_rate_BC = kwargs.setdefault('learning_rate_BC',learning_rate)
    entropy_coeff = kwargs.setdefault('entropy_coeff',0)                     
    H_dims = kwargs.setdefault('H_dims',(512,) if not is_atari else None)
    linear = kwargs.setdefault('linear',False)
    N_test_E_traj = kwargs.setdefault('N_test_E_traj',50)                       # Test dataset size
    N_test_rollout = kwargs.setdefault('N_test_rollout',50)                     # N trajs for final rollout
    batch_size = kwargs.setdefault('batch_size',128)                            # Optimization batch sizes
    reinit_opt = kwargs.setdefault('reinit_opt',False)                          # Reinitialize optimization at each agg iter or build on previous parameters
    record_intermediate = kwargs.setdefault('record_intermediate',True if not (is_fail or is_cov) else False)  #Record intermediate results in results_df
    NN_mid_as_feats = kwargs.setdefault('NN_mid_as_feats',False if is_fail else None)  #Use NN second to last layer as features for FAIL rather than observation    
    N_FAIL_samps = kwargs.setdefault('N_FAIL_samps',1 if is_fail else None)     #How many times to resample next state per expert samp
    render = kwargs.setdefault('render',False)                                  #Render all traj generation
    render_final = kwargs.setdefault('render_final',render)                     #Render final traj generation
    switch_2_E_after = kwargs.setdefault('switch_2_E_after',100 if alg=='Expert-FAIL' else None)
    random_demos = kwargs.setdefault('random_demos',False)                      #Load saved demos or generate new random ones
    RL_expert_folder = kwargs.setdefault('RL_expert_folder',None)               #Where to look for RL expert
    RL_expert_exp_id = kwargs.setdefault('RL_expert_exp_id',0)                  #0 just means take the most recent
    RL_expert_load_best = kwargs.setdefault('RL_expert_load_best',True)         #Load best vs load final. More training (pong) after max reward can minimize steps to achieve reward
    RL_expert_algo = kwargs.setdefault('RL_expert_algo',None)                   #Which algo was used to train RL expert?
    T_max_all = kwargs.setdefault('T_max_all',None)                             # Max timesteps in eval and from loaded expert
    T_max = kwargs.setdefault('T_max',T_max_all)                                # Max timesteps in eval environments
    model_reg_coeff = kwargs.setdefault('model_reg_coeff',None)                 # amount of regularization to apply to model weights/biases
    partial_obs = kwargs.setdefault('partial_obs',False)                        # Shortcut to preset partial observability hidden dims for each environment
    results_path = kwargs.setdefault('results_path',None)                       # Where to save (append) results
    recent_samp_priority_exp = kwargs.setdefault('recent_samp_priority_exp',1)  # prefers more recently generated samples with weight x**(i_now-i_generated)
    horizon_weight_offset_exp = kwargs.setdefault('horizon_weight_offset_exp',None) # horizon for rolling out learner
    pair_with_E = kwargs.setdefault('pair_with_E',True if (alg in ['DaD']) or (horizon_weight_offset_exp is not None) else False)
    drop_first = kwargs.setdefault('drop_first',0)
    adversary_t_bucket_size = kwargs.setdefault('adversary_t_bucket_size',None) 
    recompute_adversary_freq = kwargs.setdefault('recompute_adversary_freq',1)  #1 is every ALICE iter, 2 is every epoch, 3 is every step
    
    density_ratio_alpha = kwargs.setdefault('density_ratio_alpha',1)            # Softens density ratio by setting it equal to r**alpha, where 0 <= alpha <= 1.
    
    if H_dims is not None:
        H_dims = kwargs['H_dims'] = tuple(H_dims) if type(H_dims) in [list,tuple] else (H_dims,)
    if kill_feats is not None:
        kill_feats = kwargs['kill_feats'] = tuple(kill_feats) if type(kill_feats) in [list,tuple] else (kill_feats,)
    if partial_obs:
        kill_feats = {'CartPole-v1':(3),'Acrobot-v1':(4,5),'MountainCar-v0':(1),
                      'Reacher-v2':(),'Hopper-v2':(),'HalfCheetah-v2':(),'Walker2d-v2':(),'Humanoid-v2':(),'Ant-v2':()}.get(env_id,())
    max_len = {'CartPole-v1':500,'Reacher-v2':1000,'Hopper-v2':1000,'HalfCheetah-v2':1000,'Walker2d-v2':1000,'Humanoid-v2':1000,'Ant-v2':1000,
               'MountainCar-v0':200,'Acrobot-v1':200}.get(env_id,100000)


    #Get expert, train and test set
    model_E = get_zoo_model(env_id,RL_algo=RL_expert_algo,RL_expert_folder=RL_expert_folder,exp_id=RL_expert_exp_id,load_best=RL_expert_load_best)
    eid = '' if RL_expert_exp_id==0 else f'_{RL_expert_exp_id}'
    
    pi_E = lambda obs: model_E.predict(obs)[0] #model_E.predict returns action,state for recurrent policies
    #'cached_experts/'+env_id+'-train.pkl.xz' #'IL_experts/'+env_id+'_demo_trajs.pkl.xz'    #'IL_experts/'+env_id+'_validation_trajs.pkl' 
    df_E = get_trajectories(pi_E,env_id,N_traj=N_E_traj,path=(None if random_demos else 'cached_experts/'+env_id+eid+'-train.pkl.xz'),
                            T_max=T_max,verbose=verbose-2,vec_env=True,obs_postprocess=None,seed=None)
    df_test = get_trajectories(pi_E,env_id,N_traj=N_test_E_traj,path=(None if random_demos else 'cached_experts/'+env_id+eid+'-test.pkl.xz'),
                               T_max=T_max,verbose=verbose-2,vec_env=True,obs_postprocess=None,seed=None)
    if T_max_all is not None:
        df_E = df_E[df_E['t']<T_max_all]
        df_test = df_test[df_test['t']<T_max_all]
    train_rew = [np.sum(df_E[df_E['traj_ind']==i]['rew'].to_numpy()) for i in pd.unique(df_E['traj_ind'])]
    test_rew = [np.sum(df_test[df_test['traj_ind']==i]['rew'].to_numpy()) for i in pd.unique(df_test['traj_ind'])]
    kwargs.update(train_r_mean=np.mean(train_rew),train_r_std=np.std(train_rew),test_r_mean=np.mean(test_rew),test_r_std=np.std(test_rew))
    print('Train set reward {train_r_mean:.1f}+-{train_r_std:.1f}, Test set reward {test_r_mean:.1f}+-{test_r_std:.1f}'.format(**kwargs))

    sample_env = make_env(env_id) #for sampling next states
    DISCRETE = hasattr(sample_env.action_space,'n')
    loss_function = kwargs.setdefault('loss_function','mse' if not DISCRETE else None)
    #max_len = df_E['t'].max()

    #Obs_shape,A_dim = learner_pre(sample_env.reset(),env_id=env_id)[0].shape, sample_env.action_space.n
    A_dim = sample_env.action_space.n if DISCRETE else sample_env.action_space.shape[0]
    A_shape = (1,) if DISCRETE else (A_dim,)
    #Add history and kill features on expert 
    learner_pre = lambda obs,env_id,**kwargs : add_batch_dim(warp_obs(obs=expert_post(obs),env_id=env_id,add_history=add_history,kill_feats=kill_feats,A_shape=A_shape,**kwargs))
    expert_post = de_framestack if (is_atari and alg!='Expert') else (lambda obs:obs)
    Hstr = '{}{}'.format('+H' if add_history else '','-PO'+''.join([str(f) for f in kill_feats]) if kill_feats is not None else '')
    if add_history and alg=='Expert':
        raise NotImplementedError('Expert policy not trained with additional features')
    df_E[['obs','obs_next']] = df_E.apply(lambda row : (warp_obs(expert_post(row['obs']),env_id,row['action_prev'],add_history=add_history,kill_feats=kill_feats,A_shape=A_shape),
                                                 warp_obs(expert_post(row['obs_next']),env_id,row['action'],add_history=add_history,kill_feats=kill_feats,A_shape=A_shape)),axis=1,result_type='expand')
    df_test[['obs','obs_next']] = df_test.apply(lambda row : (warp_obs(expert_post(row['obs']),env_id,row['action_prev'],add_history=add_history,kill_feats=kill_feats,A_shape=A_shape),
                                                 warp_obs(expert_post(row['obs_next']),env_id,row['action'],add_history=add_history,kill_feats=kill_feats,A_shape=A_shape)),axis=1,result_type='expand')
    if 0:
        save_df_to_pkl(df_E,'IL_experts/'+env_id+'_demo_trajs_H.pkl')
        save_df_to_pkl(df_test,'IL_experts/'+env_id+'_validation_trajs_H.pkl')
   

    regression_loss = {'L2':L2_loss,'L1':L1_loss,'mse':mse_loss,'logcosh':logcosh_loss}.get(loss_function,L2_loss)
    test_loss = softmax_cross_entropy if DISCRETE else regression_loss
    classification_loss = zeroone_loss if DISCRETE else regression_loss
    train_loss = FAIL_loss if alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL'] else test_loss

    if DISCRETE:
        model2policy = lambda model: lambda obs: np.squeeze(np.argmax(model(obs),axis=-1))
        model_prob_a = lambda model_out,a: tf.gather_nd(tf.nn.softmax(model_out,axis=-1),a,batch_dims=1)
        clip_range = None
    else:
        model2policy = lambda model: lambda obs: np.squeeze(model(obs))
        model_prob_a = lambda model_out,a: 0
        clip_range = [sample_env.action_space.low,sample_env.action_space.high]

    #df_E = df_E[df_E['action_prev']==df_E['action']].reset_index(drop=True)
    #Get NN dims, partial functions, and alg-specific loss func

    Obs_shape = df_E['obs'][0].shape[1:] if (alg=='Expert' or is_atari) else add_batch_dim(df_E['obs'][0]).shape[1:]
    _keras_NN = partial(keras_NN,Obs_shape=Obs_shape,A_dim=A_dim,cnn=is_atari,linear=linear,seed=opt_seed,H_dims=H_dims,model_reg_coeff=model_reg_coeff,clip_range=clip_range)
    _train_model = partial(train_model,verbose=verbose-1,N_epoch=N_epoch,batch_size=batch_size,steps_per_epoch=int(opt_steps_per_iter/N_epoch),seed=opt_seed,learning_rate=learning_rate,df_test=df_test,test_loss=test_loss,model_prob_a=model_prob_a,entropy_coeff=entropy_coeff,recompute_adversary_freq=recompute_adversary_freq,DISCRETE=DISCRETE)
    _get_trajectories = partial(get_trajectories,env_id=env_id,N_traj=N_ALICE_traj,obs_preprocess=learner_pre,obs_postprocess=learner_pre,init_from_df=pair_with_E,
                                verbose=verbose-1,expert_after_n=switch_2_E_after,policy_e=pi_E,e_prepro=learner_pre,render=render,T_max=T_max,
                                randinit_t=horizon_weight_offset_exp is not None,choose_random_expert=(N_ALICE_traj!=N_E_traj))
    _eval_pi_lightweight = partial(eval_pi_lightweight,env_id=env_id,N_traj=N_test_rollout,run_seed=run_seed,obs_preprocess=learner_pre,verbose=verbose-1,T_max=T_max_all)


    start_time = time.time()

    #Initialize model list
    kwargs.update(alg = alg + Hstr, env_id = env_id + eid, N_E_samp = len(df_E))
    
    model_list,weights_E_list = [_keras_NN()],[np.ones((len(df_E)))]
    df_train,adversary_f,df_L = df_E,None,None
    epoch_train_losses = []
    df_E['i_agg'] = np.zeros(len(df_E))
    i_agg = 0

    if alg in ['BC','DaD','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL'] and not FORWARD:
        
        results_dicts = [copy.copy(kwargs) for i in range(N_agg_iter)]
        
        #Initialize first policy by training BC
        train_results = _train_model(model_list[-1],df_E,test_loss,learning_rate=learning_rate_BC)
        epoch_train_losses.extend(train_results['epoch_losses'])
        results_dicts[0].update(train_results)
        
        #Begin iterations
        for i_agg in range(1,N_agg_iter):
            if verbose>1:
                print('{} iter {}, {} train samps'.format(alg+Hstr, i_agg, len(df_train),))

            ### Collect and agg data
            if alg not in ['Expert-FAIL']:
                pi_i = model2policy(model_list[-1])
                N_L_prev = len(df_L) if df_L is not None else 0
                if verbose>2:
                    print('df_L pre gen',N_L_prev)
                horizon = max_len if horizon_weight_offset_exp is None else min(int(horizon_weight_offset_exp[0]*(i_agg**horizon_weight_offset_exp[2]) + horizon_weight_offset_exp[1]),max_len)
                df_L = _get_trajectories(pi_i,df_agg=df_L,df_init=df_E,horizon=horizon,seed=run_seed+1000*i_agg)
                df_L.loc[np.arange(N_L_prev,len(df_L)),'i_agg'] = np.ones(len(df_L)-N_L_prev)*i_agg
                
                df_L = df_L[df_L['t']>=drop_first]
                
                if is_fail:
                    df_L.dropna()
                if verbose>2:
                    print('df_L post gen',len(df_L),'num_new',len(df_L)-N_L_prev,'horizon',horizon)

            ### Save intermediate results
            if horizon_weight_offset_exp is not None:
                reward,reward_std = _eval_pi_lightweight(pi_i,N_traj=10)
            else:
                rewards = [np.sum(df_L[N_L_prev:][df_L['traj_ind']==i]['rew'].to_numpy()) for i in pd.unique(df_L[N_L_prev:]['traj_ind'])]
                if verbose>2:
                    print(rewards)
                reward,reward_std = np.mean(rewards),np.std(rewards)
            if verbose>2:
                print('rew/std',reward,reward_std)
            
            if i_agg>1:
                hindsight_losses = [batch_avg_loss(train_loss,model,df_train,adversary_f,model_prob_a,entropy_coeff) for model in model_list]
                best_ind = np.argmin(hindsight_losses)
            else:
                #df_train doesn't exist yet
                best_ind = 0
            
            #These are results for policy trained on previous iteration, but rolled out at the beginning of this iteration
            results_dicts[i_agg-1].update({'w_max':df_E['weight'].max(), 'w_min':df_E['weight'].min(), 'final':False, 'iteration_num':i_agg-1,'horizon':horizon,
                                          'w_ESS':np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2,
                                          'reward':reward,'reward_std':reward_std, 'total_opt_steps':opt_steps_per_iter*i_agg,
                                          'reward_curr':reward,'reward_std_curr':reward_std,
                                          'loss_test':batch_avg_loss(test_loss,model_list[-1],df_test,adversary_f,model_prob_a),
                                          'loss_train':batch_avg_loss(train_loss if i_agg>1 else test_loss,model_list[-1],df_train,adversary_f,model_prob_a,entropy_coeff),
                                          'class_test':batch_avg_loss(classification_loss,model_list[-1],df_test,adversary_f,model_prob_a),
                                          'runtime':(time.time()-start_time)/60,'best_ind':best_ind,'JS_div':js_from_samples(df_L['obs_next'].values,df_E['obs_next'].values)})
            #print(df_L['i_agg'].values)
            
            ### Set up training dataset
            #Learn density ratio weighting
            if is_cov:
                weights = estimate_ratio_on_samps(df_L['obs'].values,df_E['obs'].values,density_ratio_feature_map,warm_start=True,verbose=verbose-1)[len(df_L):]
                weights_E_list.append(weights)#/np.mean(weights))
                df_E['weight'] = w_E = np.mean(weights_E_list,axis=0)
                if verbose >1:
                    print('Weights - max {:.1f}, min {:.1f}, effective sample size {:d} ({:d})'.format(max(w_E),min(w_E),int(np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2),len(w_E)))
            
            # Adversary fitting
            if is_fail:
                # Simulate alt next states
                df_L = resample_next_states(df_L,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_L)-N_L_prev,verbose=verbose-1,obs_postprocess=learner_pre)
                if alg in ['Expert-FAIL'] and i_agg==1:
                    df_L = resample_next_states(df_E,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_E),verbose=verbose-1,obs_postprocess=learner_pre)
                    
                #Reweigh learner for recency
                #print(df_L['i_agg'].values,np.power(recent_samp_priority_exp,i_agg-df_L['i_agg'].values))
                df_L['weight'] = np.power(recent_samp_priority_exp,i_agg-df_L['i_agg'].values)
                #if verbose>1:
                #    print('min_weight',df_L['weight'].min(),'max_weight',df_L['weight'].max())
                
                #Fit adversary function
                recompute_adversary = lambda model: fit_adversary_t(df_L['obs_next_ref'].values,df_E['obs_next'].values,
                                                                  df_L['weight'].values*compute_action_prob(df_L,model,model_prob_a)/df_L['action_ref_prob'].to_numpy(),
                                                                  df_E['weight'].values,adversary_feature_map,NN_mid_as_feats,model,Obs_shape,df_L['t'].values,
                                                                  df_E['t'].values,adversary_t_bucket_size)
                adversary_f = recompute_adversary(model_list[-1])
                
            else:
                adversary_f = None
                recompute_adversary = None
            
            df_train = setup_training_dataframe(alg,df_E,df_L)
            #print('df_E',len(df_E),'df_L',len(df_L) if df_L is not None else 0,'df_train',len(df_train))
            
            ### Train
            new_model = _keras_NN() if reinit_opt else clone_model_and_weights(model_list[-1])
            train_results = _train_model(new_model,df_train,train_loss,adversary_f=adversary_f,recompute_adversary=recompute_adversary)
            results_dicts[i_agg].update(train_results)
            epoch_train_losses.extend(train_results['epoch_losses'])
            model_list.append(new_model)
            #print(df_train['loss'].min(),df_train['loss'].max(),df_train['loss'].mean())
            
    if alg in ['BC','DaD','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL'] and FORWARD:
        T_max_E = df_E['t'].max()
        FORWARD_T = T_max or T_max_E        #If max_T is None, use max_T_E
        FORWARD_H = 1
        model_t_list = []
        adversary_t_list = [] #Don't actually need to save these for every timestep since they're only used during training
        results_dicts = []
        
        # At step zero
        model_t_list.append(_keras_NN())
        adversary_t_list.append(None) 
        train_results = _train_model(model_t_list[0],df_E[df_E['t']<FORWARD_H],test_loss)
        
        df_L = _get_trajectories(model2policy(model_t_list[0]),df_agg=None,df_init=df_E,T_max=FORWARD_H)
        w_E = np.ones(len(df_E[df_E['t']<FORWARD_H]))
        reward,reward_std = FORWARD_eval_pi_lightweight(model_t_list,model2policy,env_id,FORWARD_H=FORWARD_H,N_traj=N_test_rollout,render=render_final,
                                                        run_seed=run_seed,vec_env=False,obs_preprocess=learner_pre,verbose=verbose-2,T_max=0)
        results_dicts.append(copy.copy(kwargs))
        results_dicts[0].update(train_results)
        results_dicts[0].update({'w_max':w_E.max(), 'w_min':w_E.min(), 'final':False, 'iteration_num':0,'horizon':FORWARD_H,
                                 'w_ESS':np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2,
                                 'reward':reward,'reward_std':reward_std, 'total_opt_steps':opt_steps_per_iter,
                                 'loss_test':batch_avg_loss(test_loss,model_t_list[-1],df_test[df_test['t']<FORWARD_H],None,model_prob_a),
                                 'loss_train':batch_avg_loss(test_loss,model_t_list[-1],df_E[df_E['t']<FORWARD_H],None,model_prob_a,entropy_coeff),
                                 'class_test':batch_avg_loss(classification_loss,model_t_list[-1],df_test[df_test['t']<FORWARD_H],None,model_prob_a),
                                 'runtime':(time.time()-start_time)/60,'JS_div':js_from_samples(df_L['obs_next'][df_L['t']<FORWARD_H].values,df_E['obs'][df_E['t']<=FORWARD_H].values)})
        print('t',0,'df_L',len(df_L))
        for tt in range(FORWARD_H,FORWARD_T,FORWARD_H):
            
            #df_train = s'_tm1_L,  s*_t,a*_t
            #L_tm1_inds = (df_L['t'] >= tt-FORWARD_H) & (df_L['t'] <  tt)
            #E_t_inds   = (df_E['t'] >  tt-FORWARD_H) & (df_E['t'] <= tt)
            #test_t_inds   = (df_test['t'] >  tt-FORWARD_H) & (df_test['t'] <= tt)
            
            L_tm1_inds = df_L['t'] == tt-1     # s' ~ pi_Lt-1   s,a ~ pi_Et
            E_t_inds   = df_E['t'] == tt
            test_t_inds = df_test['t'] == tt
            
            # t      s       a       s'      a_prev   weight
            # 0      s0      a0      s1      None     1
            # 1      s1      a1      s2      a0       1
            # 2      s2      a2      s3      a1       1
            # 3      s3      a3      s4      a2       1
            
            # Cov failure points
            #   Density ratio estimation is bad -- use alpha
            #           Divide by max of weights
            
            if is_cov:
                ## Calculate density ration if cov
                weights = estimate_ratio_on_samps(df_L['obs_next'][L_tm1_inds].values,df_E['obs'][E_t_inds].values,density_ratio_feature_map,warm_start=False,verbose=verbose-1)
                
                w_E = weights[sum(L_tm1_inds):]
                w_E = w_E/w_E.max()    #readjust
                w_E = w_E**density_ratio_alpha 
                #weights = weights**alpha #alpha between zero and one
                
                df_E.loc[E_t_inds,'weight'] = w_E
                df_E
                
                if verbose >1:
                    print('Weights - max {:.3f}, min {:.3f}, effective sample size {:d} ({:d})'.format(
                           max(w_E),min(w_E),int(np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2),len(w_E)))
 
            if is_fail:
                ## Simulate alt next states if fail
                df_L_t = FORWARD_sample_next_states(df_L[L_tm1_inds],sample_env,A_dim,n_samp=N_FAIL_samps,verbose=verbose-1,obs_postprocess=learner_pre)
                
                
                recompute_adversary = lambda model: fit_adversary(df_L_t['obs_next_ref'].values,df_E['obs_next'][E_t_inds].values,df_L_t['weight'].values*compute_action_prob(df_L_t,model,model_prob_a)/df_L_t['action_ref_prob'].to_numpy(),df_E['weight'][E_t_inds].values,adversary_feature_map,NN_mid_as_feats,model,Obs_shape)
                adversary_f = recompute_adversary(model_t_list[-1]) #initialize adversary using previous model
            else:
                df_L_t = None
                recompute_adversary = None
                adversary_f = None

            ## Train
            df_train = setup_training_dataframe(alg,df_E[E_t_inds],df_L_t)
            
            new_model = _keras_NN() if reinit_opt else clone_model_and_weights(model_t_list[-1])
            model_t_list.append(new_model)
            
            train_results = _train_model(model_t_list[-1],df_train,train_loss,adversary_f=adversary_f,recompute_adversary=recompute_adversary)
            #adversary_t_list.append(recompute_adversary(model_t_list[-1]))
            
            ## Step forward with trained model
            df_L = _get_trajectories(model2policy(model_t_list[-1]),df_agg=df_L,df_init=df_L,horizon=FORWARD_H,
                                     init_ts=[tt]*N_ALICE_traj,init_from_df=True,randinit_t=False,choose_random_expert=False,init_sprime=True)
            
            # Save Results
            reward,reward_std = FORWARD_eval_pi_lightweight(model_t_list,model2policy,env_id,FORWARD_H=FORWARD_H,N_traj=N_test_rollout,render=render_final,
                                                        run_seed=run_seed,vec_env=False,obs_preprocess=learner_pre,verbose=verbose-2,T_max=tt)
            results_dicts.append(copy.copy(kwargs))
            results_dicts[-1].update(train_results)
            results_dicts[-1].update({'w_max':w_E.max(), 'w_min':w_E.min(), 'final':False, 'iteration_num':tt//FORWARD_H,'horizon':FORWARD_H,
                                      'w_ESS':np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2,
                                      'reward':reward,'reward_std':reward_std, 'total_opt_steps':opt_steps_per_iter,
                                      'loss_test':batch_avg_loss(test_loss,model_t_list[-1],df_test[test_t_inds],adversary_f,model_prob_a),
                                      'loss_train':batch_avg_loss(train_loss,model_t_list[-1],df_train,adversary_f,model_prob_a,entropy_coeff),
                                      'class_test':batch_avg_loss(classification_loss,model_t_list[-1],df_test[test_t_inds],adversary_f,model_prob_a),
                                      'runtime':(time.time()-start_time)/60,'JS_div':js_from_samples(df_L['obs_next'][df_L['t']==tt-1].values,df_E['obs'][E_t_inds].values)})
                                      #Reindex df_L for JS div since we increased size
            if len(df_L[df_L['t']==tt])<=1:
                break
            print('t',tt,'df_L',len(df_L),'on_policy_reward',reward)

    if alg == 'Expert':
        model_list[0] = lambda obs: model_E.proba_step(np.reshape(obs,(-1,*Obs_shape))) if DISCRETE else model_E.predict(np.reshape(obs,(-1,*Obs_shape)))

    if alg == 'Random':
        model_list[0] = lambda obs: np.random.rand(obs.shape[0],A_dim).astype(np.single)

    if alg == 'DAgger':
        raise NotImplementedError()

    
    
    if FORWARD:
        # Score (mean over timesteps)
        train_loss = np.mean([d['loss_train'] for d in results_dicts if 'loss_train' in d])
        test_loss = np.mean([d['loss_test'] for d in results_dicts if 'loss_test' in d])
        test_class = np.mean([d['class_test'] for d in results_dicts if 'class_test' in d])
        JS_div = np.mean([d['JS_div'] for d in results_dicts if 'JS_div' in d])
        # On policy loss
        reward,reward_std = FORWARD_eval_pi_lightweight(model_t_list,model2policy,env_id,FORWARD_H=1,N_traj=N_test_rollout,render=render_final,
                                                        run_seed=run_seed,vec_env=False,obs_preprocess=learner_pre,verbose=verbose-2,T_max=T_max)
        # Print and save
        print('{} {} FORWARD_{}   train:{:.5f} test:{:.5f} reward:{:.1f}+-{:.1f} ({:.1f} m)'.format(
              env_id,N_E_traj,alg+Hstr,train_loss,test_loss,reward,reward_std,(time.time()-start_time)/60))
        results_dicts.append(copy.copy(kwargs))
        results_dicts[-1].update({'w_max':df_E['weight'].max(), 'w_min':df_E['weight'].min(), 'final':True, 'iteration_num':tt//FORWARD_H,'horizon':FORWARD_H,
                                      'w_ESS':np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2,
                                      'reward':reward,'reward_std':reward_std, 'total_opt_steps':opt_steps_per_iter,
                                      'loss_test':test_loss,'loss_train':train_loss,'class_test':test_class,
                                      'runtime':(time.time()-start_time)/60,'JS_div':JS_div})
    else:
    
        if 0 and (verbose>0):
            print('Epoch Losses: '+' '.join([f'{loss:.2g}' for loss in epoch_train_losses]))
        #df_train.sort_values('E_ind',inplace=True)
        #printlist = ['action','loss'] if alg in ['BC'] else ['action_orig','action','loss','sp_dist','E_ind',]
        #print(model_list[-1](df_train['obs'][0]))
        #print(df_train[printlist][:20])
        #print(df_train[printlist][-20:])
        #Choose the best model in the list
        hindsight_losses = [batch_avg_loss(train_loss,model,df_train,adversary_f,model_prob_a,entropy_coeff) for model in model_list]
        for i in range(len(results_dicts)):
            results_dicts[i]['hindsight_loss_train'] = hindsight_losses[i]
        best_ind = np.argmin(hindsight_losses) if alg != 'BC' else len(model_list)-1
        
        
        print('Hindsight Losses:',', '.join([f'{L:.4f}' for L in hindsight_losses]))
        
        #Score
        test_loss_val = batch_avg_loss(test_loss,model_list[best_ind],df_test,adversary_f,model_prob_a)
        pi_rollout = model2policy(model_list[best_ind])
        #for i,row in df_train[['obs','action']][:10].iterrows():
        #    print(row['action'],model_list[-1](row['obs']),pi_rollout(row['obs']))
        test_rollout_df = get_trajectories(pi_rollout,env_id,N_traj=N_test_rollout,render=render_final,obs_postprocess=learner_pre,
                                           obs_preprocess=learner_pre,verbose=verbose-2,vec_env=((alg=='Expert') and is_atari),seed=run_seed,T_max=T_max)
        if df_L is None:
            df_L = test_rollout_df

        rewards = [np.sum(test_rollout_df[test_rollout_df['traj_ind']==i]['rew'].to_numpy()) for i in range(N_test_rollout)]
        reward,reward_std = np.mean(rewards),np.std(rewards)
        reward_curr,reward_std_curr = _eval_pi_lightweight(model2policy(model_list[-1]),N_traj=N_test_rollout)
        print(passed_kwargs)
        print('{} {} {}   pi_{} train:{:.5f} test:{:.5f} reward:{:.1f}+-{:.1f} ({:.1f} m)'.format(env_id,N_E_traj,alg+Hstr,best_ind,hindsight_losses[best_ind],test_loss_val,reward,reward_std,(time.time()-start_time)/60))

        results_dicts[-1].update({'w_max':df_E['weight'].max(), 'w_min':df_E['weight'].min(), 'final':True, 'iteration_num':i_agg,'horizon':max_len,
                                  'w_ESS':np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2,
                                  'reward':reward, 'reward_std':reward_std,'loss_test':test_loss_val,'reward_curr':reward_curr, 'reward_std_curr':reward_std_curr,
                                  'loss_train':hindsight_losses[best_ind],'JS_div':js_from_samples(df_L['obs_next'].values,df_E['obs_next'].values),
                                  'class_test':batch_avg_loss(classification_loss,model_list[best_ind],df_test,adversary_f,model_prob_a),
                                  'runtime':(time.time()-start_time)/60,'best_ind':best_ind})

        for i in range(len(results_dicts)):
            results_dicts[i]['reward_hindsight_best'] = results_dicts[results_dicts[i]['best_ind']]['reward_curr']
            results_dicts[i]['reward_std_hindsight_best'] = results_dicts[results_dicts[i]['best_ind']]['reward_std_curr']

    sample_env.close()

    results_df = pd.DataFrame(results_dicts)
    if results_path is not None:
        load_agg_save_safe(results_path,results_df)
    return results_df
    
    
    # Papers
    # https://arxiv.org/pdf/1805.03328.pdf Modeling  Supervisor  Safe  Sets  for  Improving  Collaboration  inHuman-Robot  Teams
    # https://arxiv.org/pdf/2010.14876.pdf Fighting Copycat Agents inBehavioral Cloning from Observation Histories
    # https://arxiv.org/pdf/1812.03079.pdf ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst
    # https://proceedings.neurips.cc/paper/2020/file/8fdd149fcaa7058caccc9c4ad5b0d89a-Paper.pdf Causal Imitation Learningwith Unobserved Confounders
    # https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43146.pdf Machine Learning:The High-Interest Credit Card of Technical Debt
    # https://arxiv.org/pdf/1603.00448.pdf Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization
    # https://arxiv.org/pdf/1606.03476.pdf GAIL
    # https://arxiv.org/pdf/1611.03852.pdf GAIL - GCL connection paper
    # https://arxiv.org/pdf/1011.0686.pdf A Reduction of Imitation Learning and Structured Predictionto No-Regret Online Learning
    # Efficient Reductions for Imitation Learning
    # https://arxiv.org/pdf/1703.01030.pdf Deeply Aggrevated
    # 808 446 2099 or 808 283 7883 
if __name__=='__main__':
    pass
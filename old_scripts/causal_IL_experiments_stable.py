# -*- coding: utf-8 -*-
"""
Created on Wed Jun 3 13:16:04 2020

@author: Jonathan
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) #hacky to get tf quiet
warnings.filterwarnings('ignore',category=UserWarning) #hacky to get tf quiet
import gym
from gym.spaces.discrete import Discrete
import baselines
from baselines import deepq
from baselines.deepq.deepq import ActWrapper
from baselines.common import models
from baselines.common.vec_env import DummyVecEnv
import baselines.common.tf_util as U
import tensorflow as tf
to_cat = tf.keras.utils.to_categorical
import os
#tf.compat.v1.enable_eager_execution()
#This is what you need if doing non-eager execution
#obs_ph = tf.placeholder(dtype=tf.float32,shape=[in_dim]) #placeholder for NN input
#out = model(obs_ph[None])
#pi_L = lambda s: np.argmax(tf.get_default_session().run(out,feed_dict={obs_ph:s}))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Polygon
from matplotlib import animation
import pyglet
import multiprocessing as mp
from copy import deepcopy
import itertools
from functools import partial

##### Gym Wrappers
class HistoryAddWrapper(gym.Wrapper):
    r'''
    Wraps OpenAI gym environments so a_{t-1} is appended to state in np array
    '''
    def __init__(self,env):
        super().__init__(env)
        self.observation_space.shape = (self.observation_space.shape[0]+1,)
    def step(self,action):
        obs,reward,done,info = super().step(action)
        return np.hstack((obs,action)),reward,done,info
    def reset(self,an1=1):
        '''Default Initial action is 1 (i.e. neither left nor right)'''
        obs = super().reset()
        return np.hstack((obs,an1))

class ResetWrapper(gym.Wrapper):
    '''gives the ability to reset environment to arbitrary state (with history!)'''
    def reset(self,s0=None,an1=1):
        if an1 is None:
            an1 = 1
        if isinstance(self.env,HistoryAddWrapper):
            #print('Reset with action {}'.format(an1))
            obs = self.env.reset(an1)
            S = len(obs)-1
        else:
            #print('Reset without previous action')
            obs = super().reset()
            S = len(obs)
        if s0 is not None:
            #set internal state
            #print(s0,obs)
            self.unwrapped.state = s0
            if '_get_ob' in dir(self.unwrapped):
                obs[:S] = self.unwrapped._get_ob()[:S] #Don't overwrite history if we added it in reset call
            else:
                obs[:S] = np.array(s0)[:S]
        return obs

def kf(obs,kill_feats=[],original_len=None):
    '''Kill Features, but only if they haven't already been removed'''
    if original_len is not None:
        if len(obs)==original_len-len(kill_feats):
            return obs
    return np.delete(obs,kill_feats)

class FeatKiller(gym.Wrapper):
    '''Given list of feature indices, this removes them from observations'''
    def __init__(self,env,feat_inds):
        super().__init__(env)
        assert max(feat_inds)<self.observation_space.shape[0] and min(feat_inds)>=0
        self.feat_inds = feat_inds
    def step(self,action):
        obs,rew,done,info = super().step(action)
        return kf(obs,self.feat_inds),rew,done,info
    def reset(self):
        obs = super().reset()
        return kf(obs,self.feat_inds)

################################################################################ LQR Env

################################################################################ OpenAI Baselines

def load_deepq_expert_to_keras(path):
    '''hacky way to take trained TF model and steal matrices for eager execution'''
    import joblib
    loaded_params = joblib.load(os.path.expanduser(path))
    keys = ['deepq/q_func/mlp_fc0/w:0',
            'deepq/q_func/mlp_fc0/b:0',
            'deepq/q_func/action_value/fully_connected/weights:0',
            'deepq/q_func/action_value/fully_connected/biases:0',
            'deepq/q_func/action_value/fully_connected_1/weights:0',
            'deepq/q_func/action_value/fully_connected_1/biases:0']
    mats = [loaded_params[k] for k in keys]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(mats[0].shape[1],input_shape=(mats[0].shape[0],),activation='tanh'))
    model.add(tf.keras.layers.Dense(mats[2].shape[1],activation='relu'))
    model.add(tf.keras.layers.Dense(mats[4].shape[1],activation='softmax'))
    model.set_weights(mats)
    return model

def get_deepq_expert(env,path,overwrite=False,verbose=True,resume_training=False,
                     callback=None,env_mode=None,total_timesteps=300000,buffer_size=50000,
                     param_noise=True,exploration_fraction=0.5,exploration_final_eps=0.02,lr=1e-3):
    '''
    Loads expert if it exists, else, trains and saves one
    Note that the model returned by act outputs a vector of action probabilities,
    you need to convert that if you need it.
    '''
    print_freq = None if verbose == False else 10

    model = models.mlp(num_hidden=64, num_layers=1)
    if os.path.exists(path) and overwrite==False:
        #U.get_session().close(); tf.reset_default_graph()
        #act = deepq.learn(env,network=model,total_timesteps=0,reuse=True,load_path=path)
        model = load_deepq_expert_to_keras(path)
    else:
        if os.path.exists(path) and resume_training==True:
            load_path=path
        else:
            load_path=None
        #tf.compat.v1.disable_eager_execution()
        U.get_session().close()
        tf.reset_default_graph()
        deepq_act = deepq.learn(env=env, network=model, lr=lr, param_noise=param_noise,
                            total_timesteps=total_timesteps,
                            buffer_size=buffer_size,
                            exploration_fraction=exploration_fraction,
                            exploration_final_eps=exploration_final_eps,
                            print_freq=print_freq, load_path=load_path,callback=callback,
                            env_mode=env_mode)
        print("Saving model to "+path)
        deepq_act.save(path)
        model = load_deepq_expert_to_keras(path)
        
    return model

def get_snippets(env,pi,traj_E,H,N_snippets):
    '''Generate N_snippets. Start at expert state t0~U(0,T), rolling pi H steps
       returns combined list of dicts for all snippets corresponding to traj_E'''
    t0s = np.random.randint(1,len(traj_E),size=N_snippets)
    init_states = [traj_E[t0]['s'] for t0 in t0s]
    init_acts = [traj_E[t0-1]['a'] for t0 in t0s]
    return list(itertools.chain(*get_policy_trajs(env,pi,N_snippets,H=H,T=len(traj_E),t0s=t0s,init_states=init_states,init_acts=init_acts)))

def forward_roll_in(env,roll_in_model_list,pi,N_traj=1,pi_roll_out=None,T_max=1e6):
    '''Rolls in on env using each model in model_list. Pi defines the proper
       mapping from model to action pi(model,state)->action. If only_return_final_state
       is selected, then we return a list of N_traj final state samples.
       each sample is a dict.'''
    trajs = []
    rewards = []
    for n_traj in range(N_traj):
        traj = []
        obs,done,episode_rew,t = env.reset(),False,0,0
        state = env.state
        while not (done or t>T_max):
            if t<len(roll_in_model_list):
                #print('t:{},roll_in'.format(t))
                action = pi(roll_in_model_list[t],obs)
            else:
                if pi_roll_out is not None:
                    #print('t:{},roll_out'.format(t))
                    action = pi_roll_out(obs)
                else:
                    break
            new_obs, rew, done, _ = env.step(action)
            new_state = env.state
            traj.append({'o':obs,'a':action,'r':rew,'op':new_obs,'t':t,'s':state,'sp':new_state})
            state,obs = new_state,new_obs
            episode_rew,t = episode_rew + rew, t + 1
        rewards.append(episode_rew)
        trajs.append(traj)
    return trajs,np.mean(rewards)

def get_policy_trajs(env,act,N_traj=1,path=None,overwrite=False,H=1e8,T=1e8,t0s=None,
                     render=False, verbose=0,init_states = None,load_order = None,
                     init_acts= None,gif_path=None):
    '''
    Loads policy trajs if they exist, else trains and saves them. 
    Each traj is list of dicts. Returns list of trajs

        -init_states can be a list of N_traj states that you want to use as init
            vals for each traj
        -init_acts can be a list of N_traj accompanying initial previous actions
            if you are in add_history setting
        -t0s can be a list of N_traj initial times to initialize to
        -path can be a file path you wish to load trajs from, 
        -load_order can be a list of indices of which particular trajs you wish
        -gif_path can be a path where it will save a gif of all trajs. (Best to
            do N_traj=1 for that. Also, play with score_label.text
        -H and T both limit total trajectory duration, H only does so with t0s
    '''
    #Load trajs if exist
    if path is not None:
        if os.path.exists(path) and overwrite==False:
            #File exists and they don't want OW so load (at most) first N_traj
            trajs = pickle.load(open(path,'rb'))
            if len(trajs)<N_traj:
                raise NotImplementedError('Need to save more trajs to this file')
            if verbose:
                print('Loaded {} trajs from {}'.format(len(trajs),path))
            if load_order is None:
                return trajs[:N_traj]
            else:
                #print(load_order[:N_traj])
                return [trajs[i] for i in load_order[:N_traj]]
    #print(env)
    if render:
        score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=480, anchor_x='left', anchor_y='top',
                color=(255,63,63,255))
    if gif_path is not None:
        img_array = []
    trajs = []
    for n_traj in range(N_traj):
        traj = []
        if init_states is not None:
            assert len(init_states)==N_traj, "Expected %d init states, got %d" % (N_traj,len(init_states))
            if init_acts is not None:
                a_init = init_acts[n_traj]
            else:
                a_init = None
            obs,done = env.reset(init_states[n_traj],a_init), False
        else:
            obs,done = env.reset(), False
        state = env.state
        episode_rew, t = 0,0
        if t0s is not None:
            t = t0s[n_traj]
            T_max = min(t+H,T)-1
        else:
            T_max = T-1
        while not (done or t>T_max):
            #print(obs)
            obs = np.array(obs)
            action = act(obs)
            if render:
                time.sleep(0.02)
                env.render(mode='rgb_array')
                score_label.text = "Action: {: d}".format(action-1)
                score_label.draw()
                env.viewer.window.flip()
                if gif_path is not None:
                    arr = np.fromstring(pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data(), dtype=np.uint8, sep='')
                    arr = arr.reshape(env.viewer.height, env.viewer.width, 4)[::-1, :, 0:3]
                    img_array.append(arr)
            new_obs, rew, done, _ = env.step(action)
            new_state = env.state
            traj.append({'o':obs,'a':action,'r':rew,'op':new_obs,'t':t,'s':state,'sp':new_state})
            state,obs = new_state,new_obs
            t += 1
            episode_rew += rew
        if verbose>1:
            print("Episode {} reward: {}".format(n_traj,episode_rew))
        trajs.append(traj)
    if render:
        env.close()
    if gif_path is not None:
        make_gif(img_array[1:],gif_path)
    if path is not None:
        pickle.dump(trajs,open(path,'wb'))
        if verbose:
            print('Saved {} trajs to {}'.format(N_traj,path))
    return trajs
def make_gif(img_array,gif_path):
    fig = plt.figure(figsize=(img_array[0].shape[1] / 100.0, img_array[0].shape[0] / 100.0), dpi=100)
    plt.axis('off')
    fig.tight_layout()
    patch = plt.imshow(img_array[0])
    animate = lambda i: patch.set_data(img_array[i])
    gif = animation.FuncAnimation(plt.gcf(), animate, frames = len(img_array), interval=50)
    gif.save(gif_path, writer='imagemagick', fps=20)


def keras_NN(in_dim,out_dim,H1_dim=8,linear=True,seed=None):
    '''NN builder func'''
    initializer = tf.glorot_uniform_initializer(seed=seed)
    if linear:
        model=tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(out_dim,input_shape=(in_dim,),activation='softmax',kernel_initializer=initializer))
        return model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(H1_dim,input_shape=(in_dim,),activation='tanh',kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(out_dim,activation='softmax',kernel_initializer=initializer))
    return model

################################################################################ NN training code
def grad(model, inputs, outputs,loss):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
def train_model(model,inputs,outputs,loss,learning_rate,N_epoch=20,batch_size=32,
                steps_per_epoch=None,verbose=0,seed=None,delta=1e-6):
    '''trains keras model, either by taking N_epoch*steps_per_epoch optimization
       steps or until step size drops below delta'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss_results = []
    steps_per_epoch = steps_per_epoch or len(inputs) #if None, take num_samp steps
    np.random.seed(seed)
    last_loss = 1e9
    for epoch in range(N_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_inds = itertools.cycle(np.random.permutation(len(inputs)//batch_size+1)) #random shuffle inds
        n_steps = 0
        while n_steps<steps_per_epoch:
            i = next(epoch_inds)
            start = i*batch_size
            stop = i*batch_size + min(steps_per_epoch-n_steps,batch_size)
            loss_value, grads = grad(model, inputs[start:stop],
                                     outputs[start:stop],loss)
            if np.abs(last_loss-loss_value.numpy())<delta:
                #print('Converged to {:.3g}'.format(last_loss-loss_value.numpy()))
                break
            last_loss = loss_value.numpy()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            n_steps += len(inputs[start:stop])
            #print(loss_value.numpy())
        train_loss_results.append(epoch_loss_avg.result())
        if verbose>1:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch+1,epoch_loss_avg.result()))
    if verbose>0:
        print("Train Loss ({} Epochs): {:.3f}".format(epoch+1,epoch_loss_avg.result()))

def reshape_agg_data(trajs,alg='BC',pi_E=None,trajs_E=None,x_old=None,y_old=None,
                     env=None,add_history=False,reject_threshold=1e6,pair_traj_E=False,
                     N_DaD2_samp=10,kill_feats=[],original_dim=None):
    '''returns a set of data x and labels y for optimizer from trajectory traj

            -aggs with x_old and y_old if present
            -queries expert pi_E for 'DAgger' alg
            -uses enviornment simulator env for 'DaD2' algs
            -compares against specific expert traj or average expert traj from 
                trajs_E for DaD algs depending on pair_traj_E
            -rejects samples from traj that deviate too much from expert wrt reject_threshold
            -if history feat present (add_history True) then it ignores history
                feature in computing deviation
            -kills features before adding them to x/y via kill_feats and original_dim'''
    x,y = [],[]
    def dist(x1,x2): #distance between two observations. If history is a feature, then we ignore that feature in computing distance
        if add_history:
            return np.linalg.norm(x1[:-1]-x2[:-1]) 
        else:
            return np.linalg.norm(x1-x2)
    A_dim = 3 #one-hot action dimension
    K = len(trajs) #Number trajs in batch
    S_dim = len(kf(trajs[0][0]['o'],kill_feats,original_dim)) #state dimension
    if trajs_E is not None: #trajs_E must be same length as trajs if present
        assert len(trajs)==len(trajs_E)
    n_rejected,n_total = 0,0
    for k in range(K): 
        if trajs_E is not None: #truncate at shortest between traj and traj_E
            T = min(len(trajs[k]),len(trajs_E[k]))
        else:
            T = len(trajs[k])

        for t_L in range(T):
            t_E = trajs[k][t_L]['t'] #Actual time value if we are referencing expert
            n_total += 1
            if trajs_E is not None:
                if dist(kf(trajs_E[k][t_E]['o'],kill_feats,original_dim),kf(trajs[k][t_L]['o'],kill_feats,original_dim))>reject_threshold:
                    n_rejected += 1
                    continue

            x.append(kf(trajs[k][t_L]['o'],kill_feats,original_dim)) #State
            if alg=='BC':
                y.append(to_cat(trajs[k][t_L]['a'],A_dim)) #Action
            if alg=='DAgger':
                y.append(to_cat(pi_E(trajs[k][t_L]['o']),A_dim)) #Expert action @ State          
            if alg=='DaD1':
                if pair_traj_E:
                    y.append(to_cat(trajs_E[k][t_E]['a'],A_dim)) #Expert action @ t
                else:
                    y.append(np.mean([to_cat(trajs_E[k][t_E]['a'],A_dim) for k in range(K) if t_E<len(trajs_E[k])],axis=0)) #Mean over one-hot actions. Not sure if this is a good idea
            if alg in ['DaD20','DaD21']:
                if pair_traj_E: #sp_E = s_{t+1} from expert traj
                    sp_E = kf(trajs_E[k][t_E]['op'],kill_feats,original_dim)
                else:
                    sp_E = np.mean([kf(trajs_E[k][t_E]['op'],kill_feats,original_dim) for k in range(K) if t_E<len(trajs_E[k])],axis=0)
            if alg=='DaD20': #for discrete actions and deterministic env [sp(0),sp(1),...,sp(A_dim),sp_E]
                y.append([*[kf(env_P(env,trajs[k][t_L]['s'],a),kill_feats,original_dim) for a in range(A_dim)],sp_E])
            if alg=='DaD21': #samples several random actions for stochastic env [[0,sp_E][a,sp(a)][a,sp(a)]...]
                S_pad,A_pad = max(0,A_dim-S_dim),max(0,S_dim-A_dim) #need s,a vecs to be same dim for tensor to work so we pad whichever is smaller
                output = [[np.zeros((A_dim+A_pad)),np.pad(sp_E,(0,S_pad))]] #on first element, zeros is just a placeholder since we don't care about expert action
                for i in range(N_DaD2_samp):
                    a = env.action_space.sample()
                    output.append([np.pad(to_cat(a,A),(0,A_pad)),np.pad(kf(env_P(env,trajs[k][t_L]['s'],a),kill_feats,original_dim),(0,S_pad))])
                y.append(output)
    if 0: #Why does this oscillate for DaD?
        print('Kept {:.1%} of samples'.format((n_total-n_rejected)/n_total))

    if x_old is not None:
        if len(x_old)>0:
            if len(x)>0:
                x,y = np.vstack([x_old,x]),np.vstack([y_old,y])
            else:
                x,y = x_old,y_old
    else:
        x,y = np.array(x),np.array(y)
    return x,y

def env_P(env,s,a,n_pred=1):
    '''returns mean next state over n_pred (sp~P(sp|s,a)) single step predictions'''
    sps = []
    assert env is not None #For now we do this in the environment, not numerically
    for i in range(n_pred):
        obs = ResetWrapper(env).reset(s)
        assert (np.array(env.state) == np.array(s)).all(), "You need to fix the reset, Jonathan, env.state {}, s {}".format(env.state,s)
        op,_,_,_ = env.step(a)
        sps.append(op)
    return np.mean(sps,axis=0)

def online_IL(alg,pi_E_path=None,env_name=None,add_history=True,N_expert_traj=5,
              reinit_opt=True,N_opt=10000,verbose=0,N_epoch=10,set_s0=True,
              N_DaD_iter=10,linear=True,DaD_snippets=False,thresh=False,kill_feats=[],
              make_gif=False,run_seed=None,opt_seed=None,DaD_progression=False,
              hindsight_best=True):
    '''Performs IL using alg and 
        returns reward, validation error, Num expert samps, train err

        prints results with following ident code +H (history) -P (paired with expert traj)
            -S (DaD_snippets hack) -T (thresholding hack) -k### (kill feats ###)
            
        
        -gets expert from pi_E_path, loads cached demos if they exist
        -Wraps with HistoryAddWrapper if add_history
        -N_expert_traj can be int, and it will progressively run [1,2,...,N_expert_traj]
            or it can be list i.e. [3,6,8] or [10] for arbitrary numbers of demo trajs
        -reinit_opt=True creates a new model at each iteration rather than improving
            on model from previous iteration
        -set_s0=True fixes initial state to be the same as expert demo traj
            rather than random draw
        -N_DaD_iter is the number of DaD improvement iterations for a given # of
            expert trajs. We also divide N_opt by N_DaD_iter to make things fair
        -linear decides whether policy class is linear or NN (H dim set immediately below)
        -DaD_snippets is a hack where at each DaD iter we progressively generate
            longer and longer horizon snippets, initialized randomly accross E traj in time
        -thresh is another DaD hack to chuck samples that diverge too far from E traj
        -kill_feats is a list of feature indices of original observation space to kill
        -make_gif makes a gif of converged policy rollout
        -run_seed sets a seed for evaluating the policies
        -opt_seed sets the seed for generating NN weights and biases before optimization
        -DaD_progression returns results accross DaD iteration #, rather than accross N_expert_trajs
        -hindsight_best=True returns best hindsight policy on aggregate data while
            False returns policy from final iteration
        '''
    #Parse args and set hyperparameters  
    H1_dim = 16
    batch_size = 128
    learning_rate = 0.1
    reject_threshold = {True:{('MountainCar-v0'):.15,('Acrobot-v1'):.5}.get((env_name),1),
                        False:1000}[thresh]
    if alg in ['BC','DAgger']:
        set_s0,DaD_snippets,thresh=False,False,False
    alg_ident = '{}{}{}{}{}{}'.format(alg, '+H' if add_history else '', '-S' if DaD_snippets else '',
                                    '-P' if set_s0 else '','-T' if thresh else '',
                                    '-k'+''.join([str(f) for f in kill_feats]) if len(kill_feats)>0 else '')
    N_DaD = N_DaD_iter #have to keep this for DaD progression
    N_DaD_iter = 1 if alg[:3]!='DaD' else N_DaD #Force N_DaD_iter to 1 for non-dad algs
    steps_per_epoch = N_opt//N_DaD_iter
    if type(N_expert_traj) is int:
        N_expert_traj = [i for i in range(1,N_expert_traj+1)]
    else:
        if alg == 'DAgger':
            raise Exception('N_expert_traj must be int for DAgger since it hits every step along the way')
    #local_params = dict(locals()) #Have to copy this otherwise next line breaks
    #params = [(k,local_params[k]) for k in sorted(local_params.keys())]
    r,err,N,train_err = [[0]*len(N_expert_traj) for i in range(4)] #Place to stick results

    #Set up env and expert
    env = gym.make(env_name)
    e_model = get_deepq_expert(env,pi_E_path,overwrite=False)
    if add_history: #This augments state with prev action
        demo_path = 'IL_experts/'+env_name+'_demo_trajs_H.pkl'
        test_path = 'IL_experts/'+env_name+'_validation_trajs_H.pkl'
        s_dim_no_hist = len(env.reset())
        env = HistoryAddWrapper(env)
        pi_E = lambda obs: np.argmax(e_model(obs[np.newaxis,:s_dim_no_hist]))
        get_policy_trajs(env,pi_E,N_traj=100,path=demo_path)
    else:
        demo_path = 'IL_experts/'+env_name+'_demo_trajs.pkl'
        test_path = 'IL_experts/'+env_name+'_validation_trajs.pkl'
        pi_E = lambda obs: np.argmax(e_model(obs[None]))
    
    #Validation data
    #A_dim = env.action_space.n if isinstance(env.action_space,Discrete) else sum(env.action_space.shape)
    trajs_v = get_policy_trajs(env,pi_E,N_traj=10,path=test_path)
    x_v,y_v = reshape_agg_data(trajs_v,'BC')
    S_dim, A_dim = x_v.shape[1], y_v.shape[1]
    original_S_dim = S_dim
    if len(kill_feats)>0:
        env = FeatKiller(env,kill_feats)
        S_dim -= len(kill_feats)

    ##LOSS FUNCTIONS
    def DaD20_loss(model,inputs,outputs,training,f=None):
        '''inputs=s_t, model(input)=pi(a|s_t) outputs = [sp(0),sp(1),sp(2),s*_t+1]'''
        sp = tf.cast(tf.slice(outputs,[0,0,0],[-1,A_dim,-1]),tf.float32) #all possible next states from s_t
        sp_E = tf.cast(tf.squeeze(tf.slice(outputs,[0,A_dim,0],[-1,1,-1])),tf.float32) #expert s*_t+1
        sp_L = tf.linalg.matvec(sp,model(inputs),transpose_a=True) #sL_t+1 weighted by pi
        if f is None:
            f = lambda x:x
        return tf.reduce_mean(f(sp_E)-f(sp_L),axis=1)**2
    def DaD20_test_loss(pi,x,y,f=None):
        '''x = s_t, y = [sp(0),sp(1),sp(2),s*_t+1]'''
        if f is None:
            f = lambda x:x
        return np.mean(f(y[-1])-f(y[pi(x)]))**2
    def DaD21_loss(model,inputs,outputs,training,f=None):
        '''inputs=s_t, model(input)=pi(a|s_t)
           outputs = [[0,mean(sE_t+1)][a,sp(a)][a,sp(a)]...] for many random a'''
        #dims are B,N+1,2,max(S_dim,A_dim) (batch, expert then N learner predictions, action and state, data)
        a_t = tf.cast(tf.squeeze(tf.slice(outputs,[0,1,0,0],[-1,-1,1,A_dim])),tf.float32)
        N_rand_samps = a_t.shape[1]
        importance_weight = tf.reduce_sum(a_t*tf.tile(tf.expand_dims(model(inputs),1),[1,N_rand_samps,1]),axis=2)*tf.cast(A_dim,tf.float32)/tf.cast(N_rand_samps,tf.float32)
        if f is None:
            f = lambda x:x
        sp_s = tf.cast(tf.squeeze(tf.slice(outputs,[0,1,1,0],[-1,-1,1,S_dim])),tf.float32) #s_t+1 ~ P(.|a_t,s_t)
        sp_E = tf.cast(tf.squeeze(tf.slice(outputs,[0,0,1,0],[-1,1,1,S_dim])),tf.float32) #expert mean(s*_t+1)
        f_sp_L = tf.linalg.matvec(f(sp_s),importance_weight,transpose_a=True) #sL_t+1 weighted by pi
        return tf.reduce_mean(f_sp_L-f(sp_E),axis=1)**2
    def DaD21_test_loss(pi,x,y,f=None):
        '''x = s_t, y = [[0,sE_t+1][a,sp(a)][a,sp(a)]...] for many random a'''
        sE_t1 = y[0][1]
        a = pi(x)
        sL_t1 = [y[i][1] for i in range(1,len(y)) if np.argmax(y[i][0])==a]
        if len(sL_t1)==0:
            sL_t1 = env_P(env,x,a)
        else:
            sL_t1 = np.mean(sL_t1,axis=0)
        if f is None:
            f = lambda x:x
        return np.mean(f(sE_t1)-f(sL_t1))**2
    def cross_entropy(model,inputs,outputs,training,f=None):
        return tf.losses.softmax_cross_entropy(outputs,model(inputs))
    def classification_loss(pi,x,y,f=None):
        return pi(x)!=np.argmax(y)

    train_loss = {'DaD20':DaD20_loss,
                  'DaD21':DaD21_loss}.get(alg,cross_entropy)
    test_loss = {'DaD20':DaD20_test_loss,
                 'DaD21':DaD21_test_loss}.get(alg,classification_loss)

    #set up partials, this just cleans up code below by hiding default args passed
    reshape_agg_data_p = partial(reshape_agg_data, pi_E=pi_E,
                                       add_history=add_history,kill_feats=kill_feats,
                                       reject_threshold=reject_threshold,env=env,
                                       original_dim=original_S_dim)
    train_model_p = partial(train_model, loss=train_loss,learning_rate=learning_rate,
                                  N_epoch=N_epoch,batch_size=batch_size,seed=opt_seed,
                                  steps_per_epoch=steps_per_epoch,verbose=verbose-1)
    score_policy_p = partial(score_policy, loss=test_loss,kill_feats=kill_feats,
                                   original_dim=original_S_dim,verbose=verbose-1)
    choose_best_hindsight_model_p = partial(choose_best_hindsight_model,
                                            loss=test_loss,kill_feats=kill_feats,
                                            original_dim=original_S_dim,verbose=verbose-1)

    if alg in ['BC','DaD1','DaD20','DaD21']:
        for NETraj_ind in range(len(N_expert_traj)):
            t_start = time.time()
            model_list = []
            for DaD_iter in range(N_DaD_iter):
                
                if DaD_iter==0: #Do BC on first iter, gathering only expert demos
                    trajs_E = get_policy_trajs(env,pi_E,N_traj=N_expert_traj[NETraj_ind],path=demo_path) #Get expert traj
                    x,y = reshape_agg_data_p(trajs_E,alg,trajs_E=trajs_E)
                    N[NETraj_ind] = len(y)
                else: #Gather DaD trajs on subsequent iters using sim
                    if DaD_snippets:
                        H = int(1.6**(DaD_iter+2)) #A guess at a good horizon schedule
                        N_snippets = int(max([len(traj_E) for traj_E in trajs_E])/H-1)+3 #A guess at the approximate number of snippets needed
                        trajs_L = [get_snippets(ResetWrapper(env),pi_L,traj_E,H,N_snippets)[:len(traj_E)] for traj_E in trajs_E]
                        #print('H {}, N_snip {}, N_samp_L {}'.format(H,N_snippets,[len(traj) for traj in trajs_L]))
                        pair_traj_E = True
                    else:
                        if set_s0:
                            init_states = [traj[0]['s'] for traj in trajs_E]
                            pair_traj_E = True
                        else:
                            init_states = None
                            pair_traj_E = False
                        trajs_L = get_policy_trajs(ResetWrapper(env),pi_L,N_traj=N_expert_traj[NETraj_ind],init_states = init_states)
                    #avg_div = avg_traj_state_divergence(trajs_L,trajs_E,add_history=add_history)
                    #print('Learner Nonzero %: {:.1%}, Avg div {:.3f}, Max div {:.2f}'.format(np.count_nonzero(avg_div)/len(avg_div),np.mean(avg_div),max(avg_div)))
                    #print(avg_div) #Useful. DaD1 with s0 diverges monotonically, DaD1 without does not.
                    x,y = reshape_agg_data_p(trajs_L,alg,trajs_E=trajs_E,x_old=x,y_old=y,pair_traj_E=pair_traj_E)

                #Retrain model
                if reinit_opt:
                    model = keras_NN(S_dim,A_dim,H1_dim,linear=linear,seed=opt_seed)
                train_model_p(model,x,y)
                model_list.append(model)
                pi_L = lambda obs: np.argmax(model(obs[None]))
                
                if verbose-2>0:
                    print('     State     a_L     pi_L')
                    for k in range(len(x)):
                        s = x[k]
                        p_L = model(s[None])[0].numpy()
                        print('[{}] [{}] [{:.2f} {:.2f} {:.2f}]'.format(' '.join(['{:.2f}'.format(si) for si in s]),pi_L(s),*p_L))
            
            env.seed(run_seed)         
            best_DaD_ind = choose_best_hindsight_model_p(model_list,env,x,y) if hindsight_best else len(model_list)-1
            min_elapsed = (time.time()-t_start)/60
            result_string = '{}, {} E_trajs, pi_{} ({:.2f}m)'.format(alg_ident,N_expert_traj[NETraj_ind],best_DaD_ind,min_elapsed)
            pi_L = lambda obs: np.argmax(model_list[best_DaD_ind](obs[None]))
            _,train_err[NETraj_ind] = score_policy_p(pi_L,None,x,y)
            r[NETraj_ind], err[NETraj_ind] = score_policy_p(pi_L,env,x_v,y_v,verbose=1,policy_name=result_string)
            
    if alg == 'DAgger':
        t_start = time.time()
        model_list = []
        for dagger_iter in range(len(N_expert_traj)):
            t_start = time.time()
            #Get on-policy states
            if dagger_iter == 0:
                trajs_E = get_policy_trajs(env,pi_E,N_traj=1,path=demo_path)
                x,y = reshape_agg_data_p(trajs_E,'BC')
            else:
                env.seed(run_seed)
                trajs_L = get_policy_trajs(env,pi_L,N_traj=1)
                x,y = reshape_agg_data_p(trajs_L,alg,x_old=x,y_old=y)
            #Retrain model
            if reinit_opt:
                model = keras_NN(S_dim,A_dim,H1_dim,linear=linear,seed=opt_seed)
            train_model_p(model,x,y)
            model_list.append(model)
            #Evaluate model
            env.seed(run_seed)
            best_DAgger_ind = choose_best_hindsight_model_p(model_list,env,x,y) if hindsight_best else len(model_list)-1
            result_string = '{} {} E_trajs pi_{} ({:.2f}m)'.format(alg_ident,N_expert_traj[dagger_iter],best_DAgger_ind,(time.time()-t_start)/60)
            pi_L = lambda obs: np.argmax(model_list[best_DAgger_ind](obs[None]))
            _,train_err[dagger_iter] = score_policy_p(pi_L,None,x,y)
            r[dagger_iter], err[dagger_iter] = score_policy_p(pi_L,env,x_v,y_v,verbose=1,policy_name=result_string)
            N[dagger_iter] = len(y)
    if alg == 'FAIL':
        #Making fail work:
        #Time-dependant policies - TimeWrapper
        #Adversarial bit. Get an LP solver
        pass
    if alg == 'FORWARD':
        for NETraj_ind in range(len(N_expert_traj)):
            H = 200#max([len(traj) for traj in trajs_v])  #pick a good horizon length
            model_seq = []
            rand_pi = lambda obs: env.action_space.sample()
            pi = lambda model,obs: np.argmax(model(obs[None]))
            for h in range(H):
                trajs_L,_ = forward_roll_in(env,model_seq,pi,N_traj=N_expert_traj[NETraj_ind],pi_roll_out=rand_pi,T_max=h)
                si_set = [traj[-1] for traj in trajs_L]
                #si_set = [traj[h] for traj in trajs_L if h<len(traj)] #Should actually be this, but haven't decided what to do in case when traj terminates before h
                #print([len(traj) for traj in trajs_L])
                x,y = reshape_agg_data_p([si_set],alg='DAgger')
                model_seq.append(keras_NN(S_dim,A_dim,H1_dim,linear,opt_seed))
                train_model_p(model_seq[h],x,y)
                _,tr_err = score_policy_p(lambda obs:np.argmax(model_seq[h](obs[None])),None,x,y,policy_name=alg+str(h))
            _,avg_r = forward_roll_in(env,model_seq,pi,N_traj=10,pi_roll_out=pi_E)
            
            N[NETraj_ind] = H*N_expert_traj[NETraj_ind]
            r[NETraj_ind] = avg_r
            print('{} {} traj, r: {:.1f}'.format(alg,N_expert_traj[NETraj_ind],avg_r))
    if make_gif:
        get_policy_trajs(env,pi_L,N_traj=1,render=True,gif_path=alg_ident+'.gif')
    if DaD_progression:
        if alg in ['DaD1','DaD20','DaD21']:
            return scores,val_errs,[e for e in range(N_DaD_iter)],DaD_errs
        else:
            return [r[0]]*N_DaD,[err[0]]*N_DaD,[e for e in range(N_DaD)],[train_err[0]]*N_DaD
    return r,err,train_err,N

def choose_best_hindsight_model(model_list,env,x,y,loss,kill_feats,original_dim,verbose=1):
    '''given list of models, returns index of model which performs best on
        dataset x,y according to loss. Only needs env if verbose'''
    if len(model_list) == 1:
        return 0
    hindsight_err = [0]*len(model_list)
    rewards = []
    for model_ind in range(len(model_list)):
        pi_L = lambda obs: np.argmax(model_list[model_ind](obs[None]))
        _,hindsight_err[model_ind] = score_policy(pi_L,None,x,y,verbose=False,loss=loss,kill_feats=kill_feats,original_dim=original_dim)
        if verbose>1:
            rewards[model_ind],_ = score_policy(pi_L,env,None,None,verbose=False)
    best_ind = np.argmin(hindsight_err)
    if verbose>=1:
        print('Validation errs: ({} best)'.format(best_ind)+', '.join(['{:.3f}'.format(e) for e in hindsight_err]))
    if verbose>1:
        print('Average reward: ({} best)'.format(np.argmax(rewards))+', '.join(['{:.3f}'.format(e) for e in rewards]))
    return best_ind
    
def avg_traj_state_divergence(diff_traj,ref_traj,add_history=False):
    '''given two sets of trajectories, returns the average state-wise divergence
        between the two'''
    assert len(diff_traj) == len(ref_traj), "Traj sets must contain same number"
    N_traj = len(ref_traj)
    T = min([len(ref_traj[i]) for i in range(N_traj)])
    dist = lambda x1,x2: np.linalg.norm(np.array(x1)-np.array(x2))
    avg_err = np.zeros((T))
    S = len(ref_traj[0][0]['o'])
    if add_history:
        S = S-1
    for n in range(N_traj):
        traj_err = np.zeros((T))
        for ind in range(len(diff_traj[n])):
            t = diff_traj[n][ind]['t']
            if t>=T:
                continue
            traj_err[t] = dist(diff_traj[n][ind]['s'][:S],ref_traj[n][t]['s'][:S])
        avg_err += traj_err/N_traj
    return avg_err

def score_policy(act,env=None,x=None,y=None,N_traj=10,verbose=True,policy_name='',loss=None,kill_feats=[],original_dim=None):
    '''scores policy by either computing validation loss or policy reward,
        depending on which of x,y or env is present'''
    if env is not None:
        trajs = get_policy_trajs(env,act,N_traj=N_traj)
        avg_reward = np.mean([sum([s['r'] for s in traj]) for traj in trajs])
        ar_str = 'avg reward: {:.1f}'.format(avg_reward)
    else:
        avg_reward = 0
        ar_str = ''
    if x is not None and y is not None:
        n_samp = len(y)
        if loss is None:
            cl_err = np.mean([act(kf(x[i],kill_feats,original_dim))!=np.argmax(y[i]) for i in range(n_samp)])
        else:
            losses = [loss(act,kf(x[i],kill_feats,original_dim),y[i]) for i in range(n_samp)]
            cl_err = np.mean(losses)
            #print(losses)
        cl_str = 'test err: {:.2f}'.format(cl_err)
    else:
        cl_err = 1.0
        cl_str = ''
    if verbose>0:
        print('Policy {} - {}, {}'.format(policy_name,cl_str,ar_str))
    return avg_reward,cl_err

def add_history(load_path,save_path):
    '''given saved trajectories, add previous history feature to that set and
        save them in save_path. Useful for ensuring that all algs access identical
        traj set regardless of whether they're in history env or not'''
    trajs = pickle.load(open(load_path,'rb'))
    for traj in trajs:
        traj[0]['o'] = np.hstack((traj[0]['o'],1))
        traj[0]['op'] = np.hstack((traj[0]['op'],traj[0]['a']))
        for t in range(1,len(traj)):
            traj[t]['o'] = np.hstack((traj[t]['o'],traj[t-1]['a']))
            traj[t]['op'] = np.hstack((traj[t]['op'],traj[t]['a']))
    pickle.dump(trajs,open(save_path,'wb'))

if __name__=='__main__':

#### Things you might want to do (in the order you might want to do them)
    if 1: #Generate a new expert and try it on the environment
        env_name = ['MountainCar-v0','Acrobot-v1'][0]
        pi_E_path = env_name+'_deepq_expert.pkl'
        env = gym.make(env_name) 
        #e_model = get_deepq_expert(env,pi_E_path,overwrite=False,total_timesteps=300000,buffer_size=50000,param_noise=False,
        #             exploration_fraction=0.5,exploration_final_eps=0.02,lr=1e-3)
        e_model = get_deepq_expert(env,pi_E_path,overwrite=False,total_timesteps=500000,buffer_size=50000,param_noise=False,
                     exploration_fraction=0.2,exploration_final_eps=0.02,lr=1e-3)
        pi_E = lambda obs: np.argmax(e_model(obs[None])[0])
        get_policy_trajs(env,pi_E,N_traj=1,render=True)
    if 0:
        #Generate and save expert trajs and add history to them
        env_name = ['MountainCar-v0','Acrobot-v1'][1]
        pi_E_path = 'IL_experts/'+env_name+'_deepq_expert.pkl'
        env = gym.make(env_name)
        e_model = get_deepq_expert(env,pi_E_path,overwrite=False)
        pi_E = lambda obs: np.argmax(e_model(obs[None])[0])
        get_policy_trajs(env,pi_E,N_traj=100,render=False,overwrite=False,path='IL_experts/'+env_name+'_demo_trajs.pkl')
        get_policy_trajs(env,pi_E,N_traj=10,render=False,overwrite=False,path='IL_experts/'+env_name+'_validation_trajs.pkl')
        add_history('IL_experts/'+env_name+'_demo_trajs.pkl','IL_experts/'+env_name+'_demo_trajs_H.pkl')
        add_history('IL_experts/'+env_name+'_validation_trajs.pkl','IL_experts/'+env_name+'_validation_trajs_H.pkl')
    if 0:
        #Try running a couple IL methods (DaD1 works best with thresh, DaD2 not working yet. There are two versions of DaD2, Dad20,Dad21, 20 works better, neither really work.)
        env_name = ['MountainCar-v0','Acrobot-v1'][1]
        pi_E_path = env_name+'_deepq_expert.pkl'
        online_IL('BC',pi_E_path,env_name,add_history=False,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True)
        online_IL('BC',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True)
        online_IL('DaD1',pi_E_path,env_name,add_history=False,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True,thresh=True)
        online_IL('DAgger',pi_E_path,env_name,add_history=False,N_expert_traj=4,N_opt=20000,N_epoch=3,verbose=0,linear=True)
    if 0:
        #Do a big batch with multiprocessing and average over opt_seed or run_seed
        env_name = ['MountainCar-v0','Acrobot-v1'][1]
        pi_E_path = 'IL_experts/'+env_name+'_deepq_expert.pkl'
        results_path = env_name+'_insert-experiment-name-here_results.pkl'
        N_expert_traj = 10
        N_opt = 20000
        N_epoch = 5
        verbose = 0
        reinit_opt = True
        N_DaD_iter = 10
        linear = False
        #Arg order for online_IL: (alg,pi_E_path,env_name,add_history,N_expert_traj,
        #   reinit_opt,N_opt,verbose,N_epoch,set_s0,N_DaD_iter,linear,DaD_snippets,
        #   thresh,kill_feats,make_gif,run_seed,opt_seed,DaD_progression,hindsight_best)
        runlist = [['BC',pi_E_path,env_name,False,4,reinit_opt,N_opt,verbose,N_epoch,False,N_DaD_iter,linear,False,False,[],False,0,0,False,True],
                   ['BC',pi_E_path,env_name,True,4,reinit_opt,N_opt,verbose,N_epoch,False,N_DaD_iter,linear,False,False,[],False,0,0,False,True],
                   ['DaD1',pi_E_path,env_name,True,4,reinit_opt,N_opt,verbose,N_epoch,False,N_DaD_iter,linear,False,False,[],False,0,0,False,True]]
            #Be sure to set your 'labels' to reflect what you put here 
        N_avg = 2
        run = False
        plot = True #Useful if you just want to plot saved run
        plt_style = ['errorbar','polygon'][1]
        multiprocess = True
        N_workers = len(runlist)*N_avg

        if run:
            results = []
            if multiprocess:
                starmaps = [0]*len(runlist)
                with mp.Pool(N_workers) as p:
                    #send jobs
                    for i in range(len(runlist)):
                        avg_runlist = []
                        for jj in range(N_avg):
                            avg_runlist.append([e for e in runlist[i]])
                            avg_runlist[-1][17] = jj #16=j for setting opt_seed, 17=j for setting run_seed
                        starmaps[i] = p.starmap_async(online_IL,avg_runlist)
                    #collect_results
                    for i in range(len(runlist)):
                        results.append(starmaps[i].get())
                        #os.system('spd-say "Done with {} {}"'.format(runlist[i][0],i))
            else:
                for args in runlist:
                    alg_results = []
                    for i in range(N_avg):
                        r,err,train_loss,Nsamp = online_IL(*args)
                        alg_results.append((r,err,train_loss,Nsamp))
                    results.append(alg_results)
            #os.system('spd-say "Finally finished!"')
            pickle.dump(results,open(results_path,'wb'))
        if plot:
            results = pickle.load(open(results_path,'rb'))
            labels = ['BC','BC+H','DaD1+H']#θ2   
            #cs = ['green','mediumseagreen','blue','steelblue','red','indianred','orange','goldenrod']
            cs = ['mediumseagreen','steelblue','indianred','goldenrod']
            #lss = ['--','-','--','-','--','-','--','-']
            lss = ['-','-','-','-']
            figs = ['On-Policy Reward','Validation Performance','Training Loss']
            xlabels = ['Number of Expert Samples']*4
            ylabels = ['On-Policy Reward','Classification Error','Training Loss']
            Nmax = 9
            for j in [0,1,2]:
                plt.figure(figs[j])
                for i in range(len(results)):
                    y = np.mean([r_set[j] for r_set in results[i]],axis=0)[:Nmax]
                    yerr = np.std([r_set[j] for r_set in results[i]],axis=0)[:Nmax]
                    N = np.mean([r_set[3] for r_set in results[i]],axis=0)[:Nmax]
                    if plt_style=='errorbar':
                        plt.errorbar(N,y,yerr=yerr,c=cs[i],ls=lss[i],label=labels[i])
                    if plt_style=='polygon':
                        plt.plot(N,y,c=cs[i],ls=lss[i],label=labels[i],lw=3)
                        xy = np.hstack((np.vstack((N,y + yerr)),np.fliplr(np.vstack((N,y - yerr))))).T
                        plt.gca().add_patch(Polygon(xy=xy, closed=True, fc=cs[i], lw=2,alpha=.2))
                #if j==0:
                    #plt.plot(N,[-90]*len(N),c='purple',label='Expert')
                plt.xlabel(xlabels[j])
                plt.ylabel(ylabels[j])
                plt.title(env_name + ' ' +figs[j])
                plt.legend()
            plt.show()


#if __name__=='__main__':
    #main()
    #run_all('MountainCar-v0')


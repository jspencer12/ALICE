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
from baselines import deepq
from baselines.common import models
import baselines.common.tf_util as U
import tensorflow as tf
import os
tf.compat.v1.enable_eager_execution()
#This is what you need if doing non-eager execution
#obs_ph = tf.placeholder(dtype=tf.float32,shape=[in_dim]) #placeholder for NN input
#out = model(obs_ph[None])
#pi_L = lambda s: np.argmax(tf.get_default_session().run(out,feed_dict={obs_ph:s}))
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import re
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
import pydotplus
import density_ratios
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics
from scipy.special import softmax

##### Consts
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

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
    def reset(self,a_n1=1):
        '''Default Initial action is 1 (i.e. neither left nor right)'''
        obs = super().reset()
        return np.hstack((obs,a_n1))

class ResetWrapper(gym.Wrapper):
    '''gives the ability to reset environment to arbitrary state (with history!)'''
    def reset(self,s0=None,a_n1=None):                                              #Action here is int, keep as one-hot
        if a_n1 is None:
            a_n1 = 1
        if isinstance(self.env,HistoryAddWrapper):
            obs = self.env.reset(a_n1)
            S = len(obs)-1
        else:
            obs = super().reset()
            S = len(obs)
        if s0 is not None:
            self.unwrapped.state = s0
            if '_get_ob' in dir(self.unwrapped):
                obs[:S] = self.unwrapped._get_ob()[:S] #Don't overwrite history if we added it in reset call
            else:
                obs[:S] = np.array(s0)[:S]
        return obs
class DiffWrapper(gym.Wrapper):
    '''Returns difference of observations'''
    def reset(self):
        obs = super().reset()
        self.last_obs = obs
        return obs-self.last_obs

def feat_func(obs,kill_feats=[],Obs_dim=None):
    if Obs_dim is not None:
        if len(obs)==original_len-len(kill_feats):
            return obs
    return np.delete(obs,kill_feats)

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

def get_expert(env,path,**kwargs):
    pass

def get_deepq_expert(env,path,overwrite=False,verbose=True,resume_training=False,
                     callback=None,env_mode=None,total_timesteps=300000):
    '''
    Loads expert if it exists, else, trains and saves one
    Note that the model returned by act outputs a vector of action probabilities,
    you need to convert that if you need it.
    '''
    if type(env) is str: #gave env_name rather than env
        if env_mode=='Hist':
            env = HistoryAddWrapper(gym.make(env))
        else:
            env = gym.make(env)
    print_freq = None if verbose == False else 10

    model = models.mlp(num_hidden=64, num_layers=1)
    if os.path.exists(path) and overwrite==False:
        #U.get_session().close(); tf.reset_default_graph()
        #act = deepq.learn(env,network=model,total_timesteps=0,reuse=True,load_path=path)
        act = load_deepq_expert_to_keras(path)
    else:
        if os.path.exists(path) and resume_training==True:
            load_path=path
        else:
            load_path=None
        tf.compat.v1.disable_eager_execution()
        U.get_session().close()
        tf.reset_default_graph()
        deepq_act = deepq.learn(env=env, network=model, lr=1e-3, param_noise=True,
                            total_timesteps=total_timesteps,
                            buffer_size=50000,
                            exploration_fraction=0.5,
                            exploration_final_eps=0.02,
                            print_freq=1, load_path=load_path,callback=callback,
                            env_mode=env_mode)
        print("Saving model to "+path)
        deepq_act.save(path)
        act = load_deepq_expert_to_keras(path)
    return act

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


def keras_NN(S_dim,A_dim,H_dims=[8],linear=True,seed=None,discrete_A=True,n_components=1):
    '''NN builder func
        Discrete out_dim is (A_dim), continuous out_dim is (2*A_dim+1 x n_components) flattend into one dimension, [component_w_1, mean_1, std_1,component_w_2,mean_2,std_2,...]
    '''
    initializer = tf.glorot_uniform_initializer(seed=seed)
    model=tf.keras.Sequential()
    if discrete_A:
        if linear:
            model.add(tf.keras.layers.Dense(A_dim,input_shape=(S_dim,),activation='softmax',kernel_initializer=initializer, bias_initializer=initializer))
            return model
        else:
            model.add(tf.keras.layers.Dense(H_dims[0],input_shape=(S_dim,),activation='tanh',kernel_initializer=initializer, bias_initializer=initializer))
            for i in range(1,len(H_dims)):
                model.add(tf.keras.layers.Dense(H_dims[i],activation='tanh',kernel_initializer=initializer, bias_initializer=initializer))
            model.add(tf.keras.layers.Dense(A_dim,activation='softmax',kernel_initializer=initializer))
            return model
    else:
        inp = tf.keras.Input(shape=(S_dim,))
        AA_dim = 1 + 2*A_dim
        component_outs = []
        for i in range(n_components):
            if linear:
                component_outs.append(tf.expand_dims(tf.keras.layers.Dense(AA_dim,kernel_initializer=initializer, bias_initializer=initializer)(inp),axis=-1))
            else:
                hidden = tf.keras.layers.Dense(H_dims[0],activation='tanh',kernel_initializer=initializer, bias_initializer=initializer)(inp)
                for i in range(1,len(H_dims)):
                    hidden = tf.keras.layers.Dense(H_dims[i],activation='tanh',kernel_initializer=initializer, bias_initializer=initializer)(hidden)
                component_outs.append(tf.expand_dims(tf.keras.layers.Dense(AA_dim,kernel_initializer=initializer, bias_initializer=initializer)(hidden),axis=-1))
        if len(component_outs) > 1:
            out = tf.keras.layers.Concatenate()(component_outs)
        else:
            out = component_outs[0]
        model = tf.keras.Model(inputs=inp,outputs=out)
        return model
        

################################################################################ NN training code
def grad(model, inputs, outputs,loss,loss_f=None,weights_E=None,weights_samp=None):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs, training=True,f=loss_f,weights_E=weights_E,weights_samp=weights_samp)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
def train_model(model,inputs,outputs,loss,learning_rate,N_epoch=20,batch_size=32,
                steps_per_epoch=None,verbose=0,seed=None,delta=1e-6,loss_f=None,weights_E=None,weights_samp=None):
    '''trains keras model, either by taking N_epoch*steps_per_epoch optimization
       steps or until step size drops below delta'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss_results = []
    steps_per_epoch = steps_per_epoch or len(inputs) #if None, take num_samp steps
    weights_samp = np.ones(len(inputs)) if weights_samp is None else weights_samp
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
            if start==len(inputs):
                continue
            loss_value, grads = grad(model, inputs[start:stop],
                                     outputs[start:stop],loss,loss_f,weights_E,weights_samp[start:stop])
            #print(last_loss,loss_value.numpy())
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

def reshape_agg_data(trajs,env,alg='BC',pi_E=None,trajs_E=None,x_old=None,y_old=None,
                     add_history=False,reject_threshold=1e6,pair_traj_E=False,
                     N_DaD2_samp=10,kill_feats=[],original_dim=None,model_L=None):
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
    x,y= [],[]
    def dist(x1,x2): #distance between two observations. If history is a feature, then we ignore that feature in computing distance
        if add_history:
            return np.linalg.norm(x1[:-1]-x2[:-1]) 
        else:
            return np.linalg.norm(x1-x2)
    #A_dim = 3 #one-hot action dimension #TODO fix this so that it's generic for continuous space (require env?)
    #A_dim = env.action_space.n if isinstance(env.action_space,Discrete) else product(env.action_space.shape)
    if isinstance(env.action_space,Discrete):
        A_dim = env.action_space.n
        action_reshape = lambda action: tf.keras.utils.to_categorical(action,A_dim)
        action_prob = discrete_probability
        discete = True
    else:
        A_dim = env.action_space.shape[0]*2+1 #np.product(env.action_space.shape) #Only robust to vector action space (can't handle matrix+ action space)
        action_reshape = lambda action: action
        action_prob = gaussian_probability
        discrete = False
    

    if model_L is None:
        model_L = lambda s: [action_reshape(pi_E(s))]

    K = len(trajs) #Number trajs in batch
    S_dim = len(kf(trajs[0][0]['o'],kill_feats,original_dim)) #state dimension
    if trajs_E is not None: #trajs_E must be same length as trajs if present
        assert len(trajs)==len(trajs_E)
    n_rejected,n_total = 0,0
    for k in range(K): 
        if trajs_E is not None:# and alg in ['DaD1','DaD20','DaD21']: #truncate at shortest between traj and traj_E
            T = min(len(trajs[k]),len(trajs_E[k]))
        else:
            T = len(trajs[k])

        for t_L in range(T):
            
            t_E = trajs[k][t_L]['t'] #Actual time value if we are referencing expert
            n_total += 1
            if trajs_E is not None and t_E<len(trajs_E[k]):
                if dist(kf(trajs_E[k][t_E]['o'],kill_feats,original_dim),kf(trajs[k][t_L]['o'],kill_feats,original_dim))>reject_threshold:
                    n_rejected += 1
                    continue

            x.append(kf(trajs[k][t_L]['o'],kill_feats,original_dim)) #State
            if alg in ['BC','ALICE-Cov','ALICE-Cov2']:
                y.append(action_reshape(trajs[k][t_L]['a'])) #Action
            if alg=='DAgger':
                y.append(action_reshape(pi_E(trajs[k][t_L]['o']))) #Expert action @ State          
            if alg=='DaD1':
                if pair_traj_E:
                    y.append(action_reshape(trajs_E[k][t_E]['a'])) #Expert action @ t
                else:
                    y.append(np.mean([action_reshape(trajs_E[k][t_E]['a']) for k in range(K) if t_E<len(trajs_E[k])],axis=0)) #Mean over expert's one-hot actions in each traj. Not sure if this is a good idea
            if alg in ['ALICE-FAIL','ALICE-Cov-FAIL']:
                #Requires pair_traj_E
                ind_E = t_L + sum([len(trajs_E[j]) for j in range(k)])
                S_pad,A_pad,int_pad = max(0,A_dim-S_dim),max(0,S_dim-A_dim),max(S_dim-1,A_dim-1)
                x.pop()
                #switch discrete vs continuous here
                for a in range(A_dim):
                    x.append(kf(trajs[k][t_L]['o'],kill_feats,original_dim))
                    y.append([np.pad(model_L(x[-1][None])[0],(0,A_pad)), #current policy action probabilities
                              np.pad(action_reshape(a),(0,A_pad)), #action a
                              np.pad(kf(env_P(env,trajs[k][t_L]['s'],a),kill_feats,original_dim),(0,S_pad)), #next state from a
                              np.pad(kf(trajs_E[k][t_E]['op'],kill_feats,original_dim),(0,S_pad)), #expert next state
                              np.pad(kf(trajs_E[k][t_E]['o'],kill_feats,original_dim),(0,S_pad)), #expert next state
                              np.pad([ind_E],(0,int_pad))]) 
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
              hindsight_best=True,ratio_estimator='KDE',expert_noise=0,adversary_feature_map='poly 2'):
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
    H_dims = [16]
    batch_size = 64
    learning_rate = 0.01
    reject_threshold = {True:{('MountainCar-v0'):.15,('Acrobot-v1'):.5}.get((env_name),1),
                        False:1000}[thresh]
    if alg in ['BC','DAgger']:
        set_s0,DaD_snippets,thresh=False,False,False
    noise_str = '-EN{:.1f}'.format(expert_noise) if expert_noise > 0 else ''
    alg_ident = ''.join([alg,'+H' if add_history else '','-S' if DaD_snippets else '',
                         '-P' if set_s0 else '','-T' if thresh else '','-L' if linear else '-NN',
                         '-k'+''.join([str(f) for f in kill_feats]) if len(kill_feats)>0 else '',
                         '-EN{:.1f}'.format(expert_noise) if expert_noise > 0 else '',
                         '-dr{}'.format(ratio_estimator) if alg[:9]=='ALICE-Cov' else '',
                         '-afm{}'.format({'poly 2':'P2','poly 1':'P1','poly 3':'P3','RBF Kernel':'RBF','RFF':'RFF'}[adversary_feature_map]) if alg[-4:]=='FAIL' else ''])

    N_DaD = N_DaD_iter #have to keep this for DaD progression
    N_DaD_iter = 1 if alg in ['BC','DAgger'] else N_DaD #Force N_DaD_iter to 1 for non-dad algs
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
    if expert_noise>0:
        env = NoiseInjector(env,expert_noise)
    if isinstance(env.action_space,Discrete):
        discrete = True
        reshape_action = lambda model_out: discrete_policy(model_out)
        action_probability = lambda model_out,a :tf.reduce_sum(a*model_out,axis=1)
    else:
        discrete = False
        reshape_action = lambda action_vec: gaussian_policy(model_out)

    e_model = get_deepq_expert(env,pi_E_path,overwrite=False)
    
    demo_path = 'IL_experts/'+env_name+noise_str+'_demo_trajs.pkl'
    test_path = 'IL_experts/'+env_name+noise_str+'_validation_trajs.pkl'
    pi_E = lambda obs: reshape_action(e_model(obs[None]))
    get_policy_trajs(env,pi_E,N_traj=100,path=demo_path) #Precompute and save
    get_policy_trajs(env,pi_E,N_traj=10,path=test_path)
    if add_history: #This augments state with prev action
        demo_path_H = 'IL_experts/'+env_name+noise_str+'_demo_trajs_H.pkl'
        test_path_H = 'IL_experts/'+env_name+noise_str+'_validation_trajs_H.pkl'
        #add_history_to_file(demo_path,demo_path_H,overwrite=False)
        #add_history_to_file(test_path,demo_path_H,overwrite=False)
        demo_path,test_path = demo_path_H,test_path_H
        obs_dim_no_hist = len(env.reset())
        env = HistoryAddWrapper(env)
        pi_E = lambda obs: reshape_action(e_model(obs[np.newaxis,:obs_dim_no_hist]))
    
    #Validation data
    #A_dim = env.action_space.n if isinstance(env.action_space,Discrete) else product(env.action_space.shape)
    trajs_test = get_policy_trajs(env,pi_E,N_traj=10,path=test_path)
    states_test,labels_test = reshape_agg_data(trajs_test,env,alg='BC')
    Feat_dim, A_dim = states_test.shape[1], labels_test.shape[1]
    Obs_dim = Feat_dim
    if len(kill_feats)>0:
        env = FeatKiller(env,kill_feats)
        Feat_dim -= len(kill_feats)

    ##LOSS FUNCTIONS
    def FAIL_loss(model,inputs,outputs,training=False,f=None,weights_E=None,weights_samp=None):
        '''inputs=s_t, outputs = [pi_ref(a),a,sp(a),spE,sE,indE] for some arbitrary (one-hot) a
           loss is model(a) A_dim*pi(a)*f(sp(a))-f(spE)
           technically f(spE) is bias that can be ignored, but we'll leave it in'''
        #dims are B,4,max(S_dim,A_dim) (batch, (policy,action,sp,sp_E), data)
        weights_samp = tf.ones((len(inputs))) if weights_samp is None else weights_samp
        a = tf.cast(tf.squeeze(tf.slice(outputs,[0,1,0],[-1,1,A_dim]),axis=1),tf.float32)
        #pi_a_s = action_probability(model(inputs),a)
        #pi_ref = tf.cast(tf.squeeze(tf.slice(outputs,[0,0,0],[-1,1,1])),tf.float32)
        action_weight = tf.reduce_sum(a*model(inputs),axis=1)*tf.cast(A_dim,tf.float32)
        #action_weight = pi_a_s/pi_ref
        if weights_E is None:
            expert_weight = tf.ones((len(inputs)))
        else:
            expert_IDs = tf.cast(tf.slice(outputs,[0,5,0],[-1,1,1]),tf.int32)
            expert_weight = tf.cast(tf.gather_nd(weights_E,expert_IDs),tf.float32)
            #lastID,lastW = expert_IDs[0,0,0].numpy(),expert_weight[0,0].numpy()
        if f is None:
            f = lambda x: tf.norm(x,axis=1)
        sp = tf.cast(tf.squeeze(tf.slice(outputs,[0,2,0],[-1,1,Feat_dim]),axis=1),tf.float32) #s_t+1 ~ P(.|a_t,s_t)
        sp_E = tf.cast(tf.squeeze(tf.slice(outputs,[0,3,0],[-1,1,Feat_dim]),axis=1),tf.float32) #expert mean(s*_t+1)
        #print(sp.shape,f(sp).shape,importance_weight.shape,weights_E.shape)
        return tf.reduce_mean(weights_samp*(action_weight*f(sp) - expert_weight*f(sp_E)))

    def FAIL_test_loss(model,x,y,f=None,weights_E=None,weights_samp=None):
        '''x=s_t, y = [pi(s_t),a,sp(a),spE] for some arbitrary (one-hot) a
           loss is model(a) A_dim*pi(a)*f(sp(a))-f(spE)
           technically f(spE) is bias that can be ignored, but we'll leave it in'''
        weights_samp = tf.ones((len(x))) if weights_samp is None else weights_samp
        if len(y.shape)==2:
            y = tf.expand_dims(y,axis=0)
        
        a_onehot = tf.cast(tf.squeeze(tf.slice(y,[0,1,0],[-1,1,A_dim]),axis=1),tf.float32)
        #pi_inds = [[pi(x_i)] for x_i in x]
        pi_inds = tf.reshape(tf.argmax(model(tf.cast(x,tf.float32)),axis=1),[-1,1])
        action_weight = tf.gather_nd(a_onehot,pi_inds,batch_dims=1)
        
        if weights_E is None:
            expert_weight = tf.ones((len(x)))
        else:
            expert_IDs = tf.cast(tf.slice(y,[0,5,0],[-1,1,1]),tf.int32)
            expert_weight = tf.cast(tf.gather_nd(weights_E,expert_IDs),tf.float32) 
        if f is None:
            f = lambda x: tf.norm(x,axis=1)
        sp = tf.cast(tf.squeeze(tf.slice(y,[0,2,0],[-1,1,Feat_dim]),axis=1),tf.float32)
        sp_E = tf.cast(tf.squeeze(tf.slice(y,[0,3,0],[-1,1,Feat_dim]),axis=1),tf.float32)
        nz_inds = tf.where(action_weight)
        return tf.reduce_mean(tf.gather_nd(weights_samp*(action_weight*f(sp) - expert_weight*f(sp_E)),nz_inds)).numpy()
    
    def regression_loss(model,inputs,outputs,training=False,f=None,weights_E=None,weights_samp=None):
        weights_samp = tf.ones((len(inputs))) if weights_samp is None else weights_samp
        raise NotImplementedError()
        return tf.compat.v1.losses.mean_squared_error
    def cross_entropy(model,inputs,outputs,training=False,f=None,weights_E=None,weights_samp=None):
        weights_samp = tf.ones((len(inputs))) if weights_samp is None else weights_samp
        inputs = tf.cast(inputs,tf.float32)
        return tf.compat.v1.losses.softmax_cross_entropy(outputs,model(inputs),weights=weights_samp)
    def classification_loss(model,inputs,outputs,f=None,weights_E=None,weights_samp=None):
        weights_samp = np.ones((len(inputs))) if weights_samp is None else weights_samp
        pi_inds = tf.argmax(model(tf.cast(inputs,tf.float32)),axis=1)
        label_inds = tf.argmax(outputs,axis=1)
        return tf.reduce_mean(tf.cast(tf.math.not_equal(pi_inds,label_inds),tf.float32))
    #def classification_loss(pi,x,y,f=None,weights_E=None,weights_samp=None):
    #    weights_samp = np.ones((len(x))) if weights_samp is None else weights_samp
    #    return np.mean([pi(x_i)!=np.argmax(y_i) for (x_i,y_i) in zip(x,y)]*weights_samp)          ######################## TODO this should generalize to non-categorical

    if discrete:
        default_train_loss = cross_entropy
        default_test_loss = cross_entropy
    else:
        default_train_loss = regression_loss
        default_test_loss = regression_loss

    train_loss = {'ALICE-FAIL':FAIL_loss,
                  'ALICE-Cov-FAIL':FAIL_loss}.get(alg,default_train_loss)
    test_loss = {'ALICE-FAIL':FAIL_loss,
                 'ALICE-Cov-FAIL':FAIL_loss}.get(alg,default_test_loss)

    ratio_estimator = {'KDE':density_ratios.KernelDensityEstimator(random_state=opt_seed),
                       'KLR':density_ratios.KernelLogisticRegression(random_state=opt_seed),
                       'PLR':density_ratios.PolynomialLogisticRegression(random_state=opt_seed,degree=int(ratio_estimator[3:]) if len(ratio_estimator)>3 else 2),
                       'VLR':density_ratios.VanillaLogisticRegression(random_state=opt_seed)}[ratio_estimator[:3]]

    #set up partials, this just cleans up code below by hiding default args passed
    reshape_agg_data_p = partial(reshape_agg_data, env=env, pi_E=pi_E,
                                       add_history=add_history,kill_feats=kill_feats,
                                       reject_threshold=reject_threshold,
                                       original_dim=Obs_dim)
    train_model_p = partial(train_model, loss=train_loss,learning_rate=learning_rate,
                                  N_epoch=N_epoch,batch_size=batch_size,seed=opt_seed,
                                  steps_per_epoch=steps_per_epoch,verbose=verbose-1,loss_f=None)
    score_policy_p = partial(score_policy, loss=test_loss,kill_feats=kill_feats,
                                   original_dim=Obs_dim,verbose=verbose-1,loss_f=None)
    choose_best_hindsight_model_p = partial(choose_best_hindsight_model,
                                            hindsight_best=hindsight_best,
                                            loss=test_loss,kill_feats=kill_feats,
                                            original_dim=Obs_dim,verbose=verbose-1,loss_f=None)

    if alg in ['BC','DaD1','DaD20','DaD21','ALICE-Cov2','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL']:
        for demo_num_ind in range(len(N_expert_traj)):
            t_start = time.time()
            model_list = []
            weights_list = []
            env.seed(run_seed)

            #Train initial (BC) policy
            trajs_E = get_policy_trajs(env,pi_E,N_traj=N_expert_traj[demo_num_ind],path=demo_path) #Get expert traj
            states_E,labels_E = reshape_agg_data_p(trajs_E,alg='BC')
            N[demo_num_ind],n_E = len(labels_E),len(labels_E)
            model = keras_NN(Feat_dim,A_dim,H_dims,linear=linear,seed=opt_seed)
            train_model_p(model,states_E,labels_E,loss=cross_entropy)
            model_list.append(model)
            pi_L = lambda obs: reshape_action(model(obs[None]))

            states_L,labels_L,weights_E,f = None,None,None,None
            weights_list.append(np.ones((len(states_E))))

            for DaD_iter in range(1,N_DaD_iter):
                
                ###### Gather data with current policy
                env.seed(run_seed)
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
                    trajs_L = get_policy_trajs(ResetWrapper(env),pi_L,N_traj=N_expert_traj[demo_num_ind],init_states = init_states)
                    #avg_div = avg_traj_state_divergence(trajs_L,trajs_E,add_history=add_history)
                    #print('Learner Nonzero %: {:.1%}, Avg div {:.3f}, Max div {:.2f}'.format(np.count_nonzero(avg_div)/len(avg_div),np.mean(avg_div),max(avg_div)))
                    #print(avg_div) #Useful. DaD1 with s0 diverges monotonically, DaD1 without does not.
                prev_num_states_L = len(states_L) if states_L is not None else 0
                states_L,labels_L = reshape_agg_data_p(trajs_L,alg=alg,trajs_E=trajs_E,x_old=states_L,y_old=labels_L,pair_traj_E=pair_traj_E,model_L = model)
                num_new_states_L = len(states_L) - prev_num_states_L
    
                ##### Maybe learn a density ratio between learner and expert samples
                if alg in ['ALICE-Cov2','ALICE-Cov','ALICE-Cov-FAIL']:
                    if alg=='ALICE-Cov':
                        state_inds = [i for i in range(len(states_L))]
                        labels = np.hstack([np.zeros((n_E)),np.ones(len(states_L))])
                    if alg in ['ALICE-Cov2','ALICE-Cov-FAIL']: #Only uses most recent samples
                        state_inds = [i for i in range(len(states_L))][-num_new_states_L:]
                        labels = np.hstack([np.zeros((n_E)),np.ones((num_new_states_L))])
                    ratio_estimator.fit(np.vstack([states_E,states_L[state_inds]]),labels,init_params='ones')
                    weights_list.append(ratio_estimator.predict_ratio(states_E))
                    weights_E = {'ALICE-Cov2':np.mean(weights_list,axis=0),
                               'ALICE-Cov':weights_list[-1],
                               'ALICE-Cov-FAIL':np.mean(weights_list,axis=0)}[alg]

                    num_params = model_list[-1].count_params()
                    #weights_E = autotune_w(weights_E,num_params*5)
                    #weights_E = weights_E/np.mean(weights_E)
                    if verbose>0:
                        
                        ESS = np.linalg.norm(weights_E,ord=1)**2/np.linalg.norm(weights_E,ord=2)**2
                        print('weight limits: ({:.5f}, {:.3f}) mean:{:.5f}, N_samp(true,eff,#param):{:d},{:.1f},{} t:{:.0f}'.format(
                              min(weights_E),max(weights_E),np.mean(weights_E),len(weights_E),ESS,num_params,time.time()-t_start))    
                        
                ##### Maybe learn an IPM metric
                if alg in ['ALICE-FAIL','ALICE-Cov-FAIL']:
                    for i in range(len(labels_L)):
                        if (states_E[int(labels_L[i,-1,0])]!=labels_L[i,-2,:Feat_dim]).any():
                            print('Error @ i:{i:d} states_E {:.2f} labels_L {:.2f}'.format(i,states_E[int(labels_L[i,-1,0]),0],labels_L[i,-2,0]))
                    f = compute_f(labels_L,Feat_dim,A_dim,iternum=DaD_iter,weights_E=weights_E,feature_map=adversary_feature_map)
                else:
                    f = None
                
                ##### Learn next policy
                if reinit_opt:
                    model = keras_NN(Feat_dim,A_dim,H_dims,linear=linear,seed=opt_seed)
                else:
                    model = tf.keras.models.clone_model(model_list[-1])

                if alg in ['ALICE-Cov','ALICE-Cov2']:
                    train_model_p(model,states_E,labels_E,weights_samp=weights_E,loss_f=f)
                else:
                    train_model_p(model,states_L,labels_L,weights_E=weights_E,loss_f=f,learning_rate=learning_rate/DaD_iter)
                model_list.append(model)
                pi_L = lambda obs: reshape_action(model(obs[None]))
                
                if verbose-3>0:
                    print('     State     a_L     pi_L')
                    for k in range(len(states_L[:10])):
                        s = states_L[k]
                        p_L = model(s[None])[0].numpy()
                        print('[{}] [{}] [{:.2f} {:.2f} {:.2f}]'.format(' '.join(['{:.2f}'.format(si) for si in s]),pi_L(s),*p_L))
            

            if alg in ['ALICE-Cov','ALICE-Cov2','BC']:
                states_train,labels_train = states_E,labels_E
            else:
                states_train,labels_train = states_L,labels_L
            #weights_E = {'ALICE-Cov2':[np.mean(weights_list[:i+1],axis=0) for i in range(len(weights_list))],
            #             'ALICE-Cov':weights_list,
            #             'ALICE-Cov-FAIL':[np.mean(weights_list[:i+1],axis=0) for i in range(len(weights_list))]}[alg]
            weights_E = [weights_E for i in range(len(weights_list))]
            #Sort out the mess of sample-wise vs indexed weights
            ws = {'ALICE-Cov':weights_E,'ALICE-Cov2':weights_E}.get(alg,None)
            we = {'ALICE-Cov-FAIL':weights_E}.get(alg,None)
            delta_inds = [i for i in range(len(states_E)) if int(states_E[i,-1])!=np.argmax(labels_E[i])]
            #print(len(delta_inds),min(weights_E[delta_inds]))
            big_inds = np.argsort(weights_E)[-10:]
            #print(weights_E[big_inds])
            #print(states_E[big_inds])
            #for ind in big_inds:
            #    print(weights_E[ind],states_E[ind],int(states_E[ind,-1]),labels_E[ind])
            #pd = sklearn.metrics.pairwise_distances(states_E[big_inds][:,:-1])
            #plt.imshow(pd)
            #plt.show()

            env.seed(run_seed)
            best_DaD_ind,train_err[demo_num_ind] = choose_best_hindsight_model_p(model_list,env,states_train,labels_train,weights_E=we,weights_samp=ws,loss_f=f)
            #pi_L = lambda obs: reshape_action(model_list[best_DaD_ind](obs[None]))
            min_elapsed = (time.time()-t_start)/60
            result_string = '{}, {} E_trajs, pi_{} ({:.2f}m)'.format(alg_ident,N_expert_traj[demo_num_ind],best_DaD_ind,min_elapsed)
            if alg in ['ALICE-Cov','ALICE-Cov2','ALICE-Cov-FAIL']:
                result_string += ' (ESS/SS {:d}/{:d})'.format(int(np.linalg.norm(weights_E[best_DaD_ind],ord=1)**2/np.linalg.norm(weights_E[best_DaD_ind],ord=2)**2),len(weights_E[best_DaD_ind]))
            r[demo_num_ind], err[demo_num_ind] = score_policy_p(model_list[best_DaD_ind],env,states_test,labels_test,verbose=1,policy_name=result_string,loss=classification_loss)
            
    if alg == 'DAgger':
        t_start = time.time()
        model_list = []
        for dagger_iter in range(len(N_expert_traj)):
            t_start = time.time()
            #Get on-policy states
            if dagger_iter == 0:
                trajs_E = get_policy_trajs(env,pi_E,N_traj=1,path=demo_path)
                states_train,labels_train = reshape_agg_data_p(trajs_E,alg='BC')
            else:
                env.seed(run_seed)
                trajs_L = get_policy_trajs(env,pi_L,N_traj=1)
                states_train,labels_train = reshape_agg_data_p(trajs_L,alg=alg,x_old=states_train,y_old=labels_train)
            #Retrain model
            if reinit_opt:
                model = keras_NN(Feat_dim,A_dim,H_dims,linear=linear,seed=opt_seed)
            train_model_p(model,states_train,labels_train)
            model_list.append(model)
            #Evaluate model
            env.seed(run_seed)
            best_DAgger_ind,train_err[dagger_iter] = choose_best_hindsight_model_p(model_list,env,states_train,labels_train)
            result_string = '{} {} E_trajs pi_{} ({:.2f}m)'.format(alg_ident,N_expert_traj[dagger_iter],best_DAgger_ind,(time.time()-t_start)/60)
            pi_L = lambda obs: reshape_action(model_list[best_DAgger_ind](obs[None]))
            r[dagger_iter], err[dagger_iter] = score_policy_p(pi_L,env,states_test,labels_test,verbose=1,policy_name=result_string)
            N[dagger_iter] = len(labels_train)

    if make_gif:
        get_policy_trajs(env,pi_L,N_traj=1,render=True,gif_path=alg_ident+'.gif')

    return r,err,train_err,N,alg_ident
def autotune_w(w,desired_ESS,tolerance=1):
    '''let w_i = w_i**alpha for some alpha<1 so that |w|_1**2/|w|_2**2~=desired_ESS'''
    ESS = lambda w : np.linalg.norm(w,ord=1)**2/np.linalg.norm(w,ord=2)**2
    scale = np.ones(len(w))
    ess = ESS(w)
    if ess>desired_ESS:
        return w
    prev_alpha = .5
    min_alpha = 0
    max_alpha = 1
    while 1:
        
        alpha_low = (prev_alpha + min_alpha)/2
        alpha_high = (prev_alpha + max_alpha)/2
        
        time.sleep(.1)
        scale_low = np.abs(w)**alpha_low/np.abs(w)
        scale_high = np.abs(w)**alpha_high/np.abs(w)
        
        ess_low = ESS(w*scale_low)
        ess_high = ESS(w*scale_high)
        if abs(desired_ESS-ess_low)<abs(desired_ESS-ess_high):
            ess,scale = ess_low,scale_low
            max_alpha,prev_alpha = alpha_high,alpha_low
        else:
            ess,scale = ess_high,scale_high
            min_alpha,prev_alpha = alpha_low,alpha_high
            
        if abs(desired_ESS-ess)<tolerance:
            return w*scale
def batch_loss(loss,model,inputs,outputs,adversary_f=None,weights_E=None,weights_samp=None,batch_size=2048):
    return sum([loss(model,inputs[i:i+batch_size],outputs[i:i+batch_size],f=adversary_f,weights_E=weights_E,weights_samp=weights_samp).numpy()*len(inputs[i:i+batch_size]) for i in range(0,len(inputs),batch_size)])/len(inputs)

def compute_f(D,Feat_dim,A_dim,gamma=None,diag=False,wen=False,max_atoms=1000,
              iternum=1,prune=True,weights_E=None,feature_map='RBF'):
    '''Given dataset D (size N), return function f that maximizes the following
        max_f sum_{s,pi(s),a,sp,sp* ~ D} A_dim*pi(a|s)f(sp)-f(sp*)      

        D = [[pi(s),one_hot(a),sp,sp*],...] where either s or a are right zero-padded
        in order to make dimensions equal

        if f(s) = w^T phi(s), where phi(s) = [k(s,s1),...,k(s,sM)] for M kernel states
        and the M kernel states are the 2*N samples of sp in D, and w is constrained
        with |w|_2 <= 1,
   '''
    N = len(D)    
    S = np.reshape(D[:,2:4,:Feat_dim].T,(Feat_dim,2*N)).T #S = [sp1,...,spN,sp*1,...,sp*N]
    expert_IDs = D[:,-1,0]

    #there are 3*iternum duplicates of each expert sample, this trims off the excess
    expert_inds = [i for i in range(N,N+N//iternum,3)]
    N_E = len(expert_inds)
    learner_inds = [i for i in range(N)]
    S = S[[i for indset in [learner_inds,expert_inds] for i in indset]]

    expert_IDs = expert_IDs[[i for i in range(0,N//iternum,3)]]
    
    phi = lambda S:S #Default
    if callable(feature_map):
        phi = feature_map
    elif type(feature_map) is str:
        rx = re.compile(r'(?P<name>\w*) *(?P<number>\d*)?')
        rxd = rx.search(feature_map).groupdict()
        if 'name' in rxd:
            if rxd['name'] == 'RBF':
                #Randomly prune down the set of atoms used for kernel func
                if prune:
                    if len(S)<max_atoms:
                        atoms = S
                    else:
                        #Try to balance classes as much as possible without expert duplicates
                        atom_inds = np.hstack((np.random.permutation(learner_inds)[:(max_atoms-min(len(expert_inds),max_atoms//2))],
                                               np.random.permutation([i for i in range(N,N+len(expert_inds))])[:max_atoms//2]))
                        atoms = S[atom_inds]
                else:
                    atoms = S

                #Choose kernel func
                if diag is False:
                    if gamma is None:
                        gamma = 1./density_ratios.median_trick(S)**2
                    kernel_func = rbf_kernel
                    #kernel = polynomial_kernel 
                else:
                    if gamma is None:
                        gamma = density_ratios.median_trick_inv_diag(X)
                    kernel_func = density_ratios.RBF_kernel_cov

                #Compute kernel mat
                K = kernel_func(S,atoms,gamma)
                phi = lambda S: kernel_func(S,atoms,gamma)
            elif rxd['name'] == 'RFF':
                N_RFF = int(rxd.get('number',1000))
                RFF_w = np.random.randn(N_RFF,S.shape[1])
                RFF_b = 2*np.pi*np.random.rand(N_RFF)
                phi = lambda S: np.cos(np.dot(S,RFF_w.T)+np.outer(len(S),RFF_b))
            elif rxd['name'] in ['poly','polynomial']:
                poly = PolynomialFeatures(int(rxd.get('number',2)))
                phi = lambda S: poly.fit_transform(S)
            else:
                print('Unknown name {}, using linear feature map'.format(rxd['name']))
        else:
            print('No name found in {}, using linear feature map'.format(rxd))
    learner_importance_weight = A_dim*np.sum(D[:,0,:A_dim]*D[:,1,:A_dim],axis=1)
    if weights_E is None:
        expert_importance_weight = np.ones((N_E))
    else:
        expert_importance_weight = np.array([weights_E[int(expert_IDs[i])] for i in range(N_E)])

    c = np.hstack([learner_importance_weight/N,-expert_importance_weight/N_E])    
    if wen and feature_map == 'RBF Kernel':
        w = c/np.sqrt(np.dot(c,phi(S)).dot(c)) 
    else:
        mu_diff = np.dot(c,phi(S))
        w = mu_diff/np.sqrt(np.inner(mu_diff,mu_diff))

    f = lambda Y: np.dot(phi(Y),w)
    return f

def gaussian_policy_helper(model_output):
    #input is a B x ((2*A+1)x K_components) tensor. First row (dim 1) is component weights,rows 1 to A+1 are mu, rows A+2 to end are log_sigma
    B,AA,K = model_output.shape
    A = (AA-1)//2
    comp_w = tf.nn.softmax(model_output[:,0,:],axis=1)
    #print(comp_w)
    mu = model_output[:,1:A+1,:]
    std = tf.exp(tf.clip_by_value(model_output[:,A+1:,:], LOG_STD_MIN, LOG_STD_MAX))
    return mu, std, comp_w
def gaussian_policy(model_output,deterministic=True):
    mu, std, comp_w = gaussian_policy_helper(model_output)
    if deterministic:
        action = tf.linalg.matvec(mu,comp_w)
    else:
        mode_choices = tf.random.categorical(tf.math.log(comp_w),1)
        action = tf.squeeze(tf.gather(mu + tf.random.normal(tf.shape(mu))*std,mode_choices,axis=2,batch_dims=1))
    if mu.shape[0] == 1:
        return action[0]
    else:
        return action
def gaussian_probability(model_output,a):
    mu, std, comp_w = gaussian_policy_helper(model_output)
    #assert mu.shape[:2] == a.shape[:2], 'Mu shape {}, a shape {}'.format(mu.shape,a.shape)
    #print([int(mu.shape[0]),int(mu.shape[1]),1])
    #print(tf.reshape(a,[int(mu.shape[0]),int(mu.shape[1]),1]))
    a = tf.cast(tf.tile(tf.reshape(a,[mu.shape[0],mu.shape[1],1]),(1,1,mu.shape[2],)),tf.float32)
    #print(mu.shape[:2],a.shape[:2],mu.numpy(),a,(mu-a).numpy(),comp_w.numpy())
    proba = tf.reduce_sum(tf.exp(tf.reduce_sum(-.5*((mu-a)/(std+EPS))**2-tf.math.log(std)-np.log(2*np.pi)/2,axis=1))*comp_w,axis=1)
    if proba.shape[0] == 1:
        return proba[0]
    else:
        return proba
    
def discrete_policy(model_output,deterministic=True):
    if deterministic:
        action = np.argmax(model_output,axis=1)
    else:
        action = model_output#np.random.choice(len(model_output),1,p=model_output)[0]
    if len(action) == 1:
        return action[0]
    else:
        return action
def discrete_probability(model_output,a):
    if type(a) is int:
        return model_output[a]
    elif a.shape == model_output.shape: #A is in one-hot
        return tf.gather_nd
        

def choose_best_hindsight_model(model_list,env,x,y,loss,kill_feats,original_dim,verbose=1,loss_f=None,weights_samp=None,weights_E=None,hindsight_best=True):
    '''given list of models, returns index of model which performs best on
        dataset x,y according to loss. Only needs env if verbose'''
    
    reshape_action = lambda action_vec: discrete_policy(action_vec) if isinstance(env.action_space,Discrete) else gaussian_policy(action_vec)

    if len(model_list) == 1 or hindsight_best==False:
        #pi_L = lambda obs: reshape_action(model_list[-1](obs[None]))
        we = weights_E[-1] if weights_E is not None else None
        ws = weights_samp[-1] if weights_samp is not None else None
        return len(model_list)-1,score_policy(model_list[-1],None,x,y,verbose=False,loss=loss,kill_feats=kill_feats,original_dim=original_dim,weights_E=weights_E,weights_samp=weights_samp,loss_f=loss_f)[1]
    hindsight_err = [0]*len(model_list)
    rewards = [0]*len(model_list)
    
    for model_ind in range(len(model_list)):
        #pi_L = lambda obs: reshape_action(model_list[model_ind](obs[None]))
        we = weights_E[model_ind] if weights_E is not None else None
        ws = weights_samp[model_ind] if weights_samp is not None else None
        _,hindsight_err[model_ind] = score_policy(model_list[model_ind],None,x,y,verbose=False,loss=loss,kill_feats=kill_feats,original_dim=original_dim,weights_E=we,weights_samp=ws,loss_f=loss_f)
        if verbose>1:
            rewards[model_ind],_ = score_policy(model_list[model_ind],env,None,None,verbose=False)
    best_ind = np.argmin(hindsight_err)
    if verbose>=1:
        print('Validation errs: ({} best)'.format(best_ind)+', '.join(['{:.3f}'.format(e) for e in hindsight_err]))
    if verbose>1:
        print('Average reward: ({} best)'.format(np.argmax(rewards))+', '.join(['{:.3f}'.format(e) for e in rewards]))
    return best_ind, hindsight_err[best_ind]
    
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

def score_policy(model,env=None,x=None,y=None,N_traj=10,verbose=True,policy_name='',loss=None,kill_feats=[],original_dim=None,loss_f=None,weights_E=None,weights_samp=None):
    '''scores policy by either computing validation loss or policy reward,
        depending on which of x,y or env is present'''
    if env is not None:
        reshape_action = lambda action_vec: discrete_policy(action_vec) if isinstance(env.action_space,Discrete) else gaussian_policy(action_vec)
        pi_L = lambda obs: reshape_action(model(obs[None]))
        trajs = get_policy_trajs(env,pi_L,N_traj=N_traj)
        avg_reward = np.mean([sum([s['r'] for s in traj]) for traj in trajs])
        ar_str = 'avg reward: {:.1f}'.format(avg_reward)
    else:
        avg_reward = 0
        ar_str = ''

    if x is not None and y is not None:
        n_samp = len(y)
        if loss is None:
            weights_samp = np.ones((len(x))) if weights_samp is None else weights_samp
            cl_err = np.mean([(np.argmax(model(kf(x[i],kill_feats,original_dim)))!=np.argmax(y[i]))*weights_samp[i] for i in range(n_samp)]) ###TODO  Score non-categorical as 
        else:
            cl_err = batch_loss(loss,model,np.array([kf(x[i],kill_feats,original_dim) for i in range(n_samp)]),y,adversary_f=loss_f,weights_E=weights_E,weights_samp=weights_samp,batch_size=2048)
        cl_str = 'test err: {:.2f}'.format(cl_err)
    else:
        cl_err = 1.0
        cl_str = ''
    if verbose>0:
        print('Policy {} - {}, {}'.format(policy_name,cl_str,ar_str))
    return avg_reward,cl_err

def add_history_to_file(load_path,save_path,overwrite):
    '''given saved trajectories, add previous history feature to that set and
        save them in save_path. Useful for ensuring that all algs access identical
        traj set regardless of whether they're in history env or not'''
    trajs = pickle.load(open(load_path,'rb'))
    if os.path.exists(save_path) and overwrite==False:
        trajs_S = pickle.load(open(save_path,'rb'))
        if len(trajs_S) == len(trajs):
            return
    for traj in trajs:
        traj[0]['o'] = np.hstack((traj[0]['o'],1))
        traj[0]['op'] = np.hstack((traj[0]['op'],traj[0]['a']))
        for t in range(1,len(traj)):
            traj[t]['o'] = np.hstack((traj[t]['o'],traj[t-1]['a']))
            traj[t]['op'] = np.hstack((traj[t]['op'],traj[t]['a']))
    pickle.dump(trajs,open(save_path,'wb'))

if __name__=='__main__':

    env_name = ['MountainCar-v0','Acrobot-v1','BreakoutNoFrameskip-v4'][1]
    pi_E_path = 'IL_experts/'+env_name+'_deepq_expert.pkl'

#### Things you might want to do (in the order you might want to do them)
    if 0: #Generate a new expert and try it on the environment
        env = gym.make(env_name) 
        def callback(lv,gv):
            if not lv['t']%100:
                print(lv['t'],len(lv['episode_rewards']),lv['episode_rewards'][-1])

        e_model = get_deepq_expert(env,pi_E_path,overwrite=False,verbose=True,callback=callback)
        pi_E = lambda obs: np.argmax(e_model(obs[None])[0])
        get_policy_trajs(env,pi_E,N_traj=1,render=True)
        #tf.keras.utils.plot_model(e_model)
    if 0:
        #Generate and save expert trajs and add history to them
        env = gym.make(env_name)
        e_model = get_deepq_expert(env,pi_E_path,overwrite=False)
        pi_E = lambda obs: np.argmax(e_model(obs[None])[0])
        get_policy_trajs(env,pi_E,N_traj=100,render=False,overwrite=False,path='IL_experts/'+env_name+'_demo_trajs.pkl')
        get_policy_trajs(env,pi_E,N_traj=10,render=False,overwrite=False,path='IL_experts/'+env_name+'_validation_trajs.pkl')
        add_history_to_file('IL_experts/'+env_name+'_demo_trajs.pkl','IL_experts/'+env_name+'_demo_trajs_H.pkl')
        add_history_to_file('IL_experts/'+env_name+'_validation_trajs.pkl','IL_experts/'+env_name+'_validation_trajs_H.pkl')
    if 0:

        #Test out gaussian policies
        ### 1-d
        model = keras_NN(1,1,H_dims=[16,8],linear=False,seed=None,discrete_A=False,n_components=5)
        model_out = model(np.zeros((100000,1)))
        mu,std,comp_w = gaussian_policy_helper(model(np.array([[0]])))
        xaxis = np.linspace(-5,5,100)
        y = np.zeros_like(xaxis)
        for i in range(mu.shape[-1]):
            plt.plot(xaxis,comp_w[0,i]/np.sqrt(2*np.pi*std[0,0,i]**2)*np.exp(-.5*(xaxis-mu[0,0,i])**2/std[0,0,i]**2))
            y += comp_w[0,i]/np.sqrt(2*np.pi*std[0,0,i]**2)*np.exp(-.5*(xaxis-mu[0,0,i])**2/std[0,0,i]**2) 
        out = gaussian_policy(model_out,deterministic=False).numpy()
        plt.hist(out,200,density=True)
        #plt.hist2d(out[:,0],out[:,1],50,density=True,range=((-5,5),(-5,5)))
        #plt.scatter(mu[:,0],mu[:,1],c='r',marker='x')
        #plt.plot(xaxis,y)
        model = keras_NN(1,2,H_dims=[16,8],linear=False,seed=None,discrete_A=False,n_components=5)
        model_out = model(np.zeros((100000,1)))
        mu,std,comp_w = gaussian_policy_helper(model(np.array([[0]])))
#        xaxis = np.linspace(-5,5,100)
#        y = np.zeros_like(xaxis)
#        for i in range(mu.shape[-1]):
#            plt.plot(xaxis,comp_w[0,i]/np.sqrt(2*np.pi*std[0,0,i]**2)*np.exp(-.5*(xaxis-mu[0,0,i])**2/std[0,0,i]**2))
#            y += comp_w[0,i]/np.sqrt(2*np.pi*std[0,0,i]**2)*np.exp(-.5*(xaxis-mu[0,0,i])**2/std[0,0,i]**2) 
        out = gaussian_policy(model_out,deterministic=False).numpy()
        #plt.hist(out,200,density=True)
        plt.hist2d(out[:,0],out[:,1],50,density=True,range=((-5,5),(-5,5)))
        plt.scatter(mu[:,0],mu[:,1],c='r',marker='x')
        #plt.plot(xaxis,y)
        plt.show()

        
    if 1:
        #Try running a couple IL methods (DaD1 works best with thresh, DaD2 not working yet. There are two versions of DaD2, Dad20,Dad21, 20 works better, neither really work.)
        #online_IL('BC',pi_E_path,env_name,add_history=False,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True,run_seed=0,opt_seed=0)
        #online_IL('BC',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True,run_seed=0,opt_seed=0)
        #online_IL('ALICE-Cov',pi_E_path,env_name,add_history=False,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=0,linear=True,set_s0=False,run_seed=0,opt_seed=0)
        #online_IL('ALICE-Cov',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=20000,N_epoch=3,verbose=3,linear=True,set_s0=True,run_seed=0,opt_seed=1,expert_noise=0,hindsight_best=True,reinit_opt=False)
        #online_IL('ALICE-Cov2',pi_E_path,env_name,add_history=True,N_expert_traj=[10],N_opt=20000,N_epoch=3,verbose=2,linear=True,set_s0=False,run_seed=0,opt_seed=0,DaD_snippets=False,thresh=False,adversary_feature_map='poly 2',ratio_estimator='PLR 1')
        #online_IL('ALICE-Cov2',pi_E_path,env_name,add_history=True,N_expert_traj=[10],N_opt=20000,N_epoch=3,verbose=2,linear=True,set_s0=False,run_seed=0,opt_seed=0,DaD_snippets=False,thresh=False,adversary_feature_map='poly 2',ratio_estimator='PLR 2')
        online_IL('ALICE-Cov-FAIL',pi_E_path,env_name,add_history=True,N_expert_traj=[10],N_opt=20000,N_epoch=3,verbose=3,linear=False,set_s0=True,run_seed=0,opt_seed=0,DaD_snippets=False,thresh=False,adversary_feature_map='poly 2',ratio_estimator='PLR 1')
        #online_IL('DaD1',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=2000,N_epoch=3,verbose=0,linear=True,thresh=True)
        #online_IL('DAgger',pi_E_path,env_name,add_history=True,N_expert_traj=2,N_opt=2000,N_epoch=3,verbose=0,linear=True)
        #online_IL('FORWARD',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=2000,N_epoch=3,verbose=0,linear=True)
        #online_IL('DaD20',pi_E_path,env_name,add_history=True,N_expert_traj=[4],N_opt=2000,N_epoch=3,verbose=0,linear=True,thresh=True)
    if 0:
        #Do a big batch with multiprocessing and average over opt_seed or run_seed
        results_paths = [env_name+'_ALICE-Cov_results.pkl',env_name+'_ALICE-C-F-CF-PLR_results.pkl']
        results_path = results_paths[0]
        N_E_traj = 10
        N_opt = 20000
        N_epoch = 5
        verbose = 0
        reinit_opt = True
        N_DaD_iter = 10
        linear = False
        #Arg order for online_IL: (alg,pi_E_path,env_name,add_history,N_expert_traj,
        #   reinit_opt,N_opt,verbose,N_epoch,set_s0,N_DaD_iter,linear,DaD_snippets,
        #   thresh,kill_feats,make_gif,run_seed,opt_seed,DaD_progression,hindsight_best,ratio_estimator,expert_noise,adversary_feature_map)
        runlist = [#['BC',            pi_E_path,env_name,False,N_E_traj,reinit_opt,N_opt,verbose,N_epoch,False,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   #['BC',            pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,False,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   #['DaD1',          pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   #['ALICE-Cov',     pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   #['ALICE-FAIL',    pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   ['ALICE-Cov2',pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'VLR',0,'poly 2'],
                   ['ALICE-Cov2',pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 2',0,'poly 2'],
                   ['ALICE-Cov2',pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 3',0,'poly 2'],
                   ['ALICE-Cov2',pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'PLR 4',0,'poly 2'],
                   ['ALICE-Cov2',pi_E_path,env_name,True, N_E_traj,reinit_opt,N_opt,verbose,N_epoch,True ,N_DaD_iter,linear,False,False,[],False,0,0,False,True,'KDE',0,'poly 3']]

            #Be sure to set your 'labels' to reflect what you put here 
        N_avg = 3
        run = False
        plot = True #Useful if you just want to plot saved run
        plt_style = ['errorbar','polygon'][1]
        multiprocess = True
        N_workers = 6#len(runlist)*N_avg

        if run:
            #if os.path.exists(results_path):
            #    results = pickle.load(open(results_path,'rb'))
            #else:
            #    results = [[] for item in runlist]
            results = [[] for item in runlist]
            if multiprocess:
                starmaps = [0]*len(runlist)
                with mp.Pool(N_workers) as p:
                    #send jobs
                    for i in range(len(runlist)):
                        avg_runlist = []
                        for jj in range(N_avg):
                            avg_runlist.append([e for e in runlist[i]])
                            avg_runlist[-1][17] = jj #16=j for setting opt_seed, 17=j for setting run_seed
                            avg_runlist[-1][16] = jj #16=j for setting opt_seed, 17=j for setting run_seed
                        starmaps[i] = p.starmap_async(online_IL,avg_runlist)
                    #collect_results
                    for i in range(len(runlist)):
                        results[i].extend(starmaps[i].get())
                        #results.append(starmaps[i].get())
                        #os.system('spd-say "Done with {} {}"'.format(runlist[i][0],i))
            else:
                for args in runlist:
                    alg_results = []
                    for i in range(N_avg):
                        r,err,train_loss,Nsamp,ident_str = online_IL(*args)
                        alg_results.append((r,err,train_loss,Nsamp,ident_str))
                    for i in range(len(runlist)):
                        results[i].extend(alg_results)
            #os.system('spd-say "Finally finished!"')
            pickle.dump(results,open(results_path,'wb'))
        if plot:
            results = [r for path in results_paths for r in pickle.load(open(path,'rb'))]
            inds = [0,1]

            labels = [results[i][0][-1] if type(results[i][0][-1])==str else str(i) for i in inds]#θ2   
            labels = ['BC','BC+H','ALICE-Cov+H','???','ALICE-FAIL+H','ALICE-Cov-FAIL+H']

            #cs = plt.rcParams['axes.prop_cycle'].by_key()['color'] #['green','mediumseagreen','blue','red','steelblue','royalblue','indianred','orange','goldenrod']
            cs = ['steelblue','steelblue','indianred','purple','indianred','indianred']
            #cs = ['mediumseagreen','steelblue','indianred','goldenrod','purple']
            lss = ['-','--',':','-.','--','-']
            #lss = ['-','-','-','-','-']
            figs = ['On-Policy Reward','Validation Performance','Training Loss']
            xlabels = ['Number of Expert Samples']*3
            ylabels = ['On-Policy Reward','Classification Error','Training Loss']
            Nmax = 10
            for j in [0]:
                plt.figure(figs[j])
                for i in inds:
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
                plt.title(env_name + ' ' +figs[j] +'\n'+results_path)
                plt.legend()
            plt.show()

# ALICE-COV
# BC loss, but weights change every time.
# 
#


###################################
# ALICE-FAIL implementation:
# x is observation, y is [[_,sp*],[a, sp(a)],[a, sp(a)],...]
# Need separate f for each t
# Ideally, for every t, I aggregate all M expert samples of sp*_t, and all N*M samples of sp_t, weighted by K*pi(a|s), then I iteratively train f,pi from these
# Step 1: collect samples, K*pi(a|s) for every t
# Step 2: train f for each t
# Step 3: train pi for all samples

#How I might do this: Mod y to have t,
# scan through data and make dict of t:{x_inds with t}
# create f for every value in dictionary
# modify f such that f(x,t) is the function to optimize over

#What gets aggregated? Try 1: aggregate nothing.
# Try 2: Aggregate states at every iteration (classes going to get imbalanced

#What is f, it is a 

#Talk - passion, humor,
#   Start with why - What's the headline you want to see in the news tomorrow?
#       Cater the technical to your audience
#       Use examples, put accomplishments in context, use stats to quantify impact
#       Two pockets - stats in one pocket, stories about people in the other pocket
#       Stories about people
#   Slides - curate images, less is more, explain everything, brief text (people are either reading or listening, not both)
#   Prepare your mind - eat food, drink, exercise, and clear your calendar before the event
#   Speaking - introduce yourself (give your credentials, tell your story in a short way so they know why you're qualified), speak slowly, 


#Notes: Feedback Meeting
# Scaling difficulty: narrow passage gaps in planning
# Our issue, control distribution is highly correlated to state distribution. Modal behavior
# System with a switching mode 
# LQR, switch @ step 
# If all expert demos are straight lines and have no recovery actions, then DaD suffers, but FAIL does not
# 

#if __name__=='__main__':
    #main()
    #run_all('MountainCar-v0')


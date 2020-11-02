import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
import gym
import time, os, itertools, sys, pickle
from scipy.special import softmax
from CIL_venv.lib.rl_baselines_zoo.utils import ALGOS, create_test_env, find_saved_model
from stable_baselines.common.tf_layers import conv_to_fc,linear
from stable_baselines.common.tf_util import make_session
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv
from stable_baselines.common.vec_env import DummyVecEnv
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,QuantileTransformer,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE
from collections import deque
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



#Eager execution
tf.compat.v1.enable_eager_execution()
#Suppress warnings
gym.logger.set_level(40)



#Constants
ATARI_ENVS = ['BeamRiderNoFrameskip-v4','BreakoutNoFrameskip-v4','EnduroNoFrameskip-v4','PongNoFrameskip-v4','QbertNoFrameskip-v4','SeaquestNoFrameskip-v4','SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4','Pong-v4']
ZOO_DIR = 'CIL_venv/lib/rl_baselines_zoo/trained_agents/'
BEST_ALGO = {'BeamRiderNoFrameskip-v4':'acktr','BreakoutNoFrameskip-v4':'acktr','EnduroNoFrameskip-v4':'dqn','PongNoFrameskip-v4':'dqn','QbertNoFrameskip-v4':'ppo2',
             'SeaquestNoFrameskip-v4':'dqn','SpaceInvadersNoFrameskip-v4':'dqn','MsPacmanNoFrameskip-v4':'acer','BipedalWalker-v2':'sac','LunarLander-v2':'dqn','LunarLanderContinuous-v2':'sac',
             'CartPole-v1':'ppo2','Acrobot-v1':'ppo2','MountainCar-v0':'acktr'}
MAX_SCORES = {'PongNoFrameskip-v4':21,'EnduroNoFrameskip-v4':700}
EPS = 1e-9

#Environment mod methods

def make_env(env_id,framestack_env=False):
    if framestack_env:
        return create_test_env(env_id, is_atari=(env_id in ATARI_ENVS), log_dir=None,) #Use this because it does nice preprocessing
    if env_id in ATARI_ENVS:
        #return ClipRewardEnv(WarpFrame(NoopResetEnv(gym.make(env_id), noop_max=30)))
        #return ClipRewardEnv(WarpFrame(MaxAndSkipEnv(NoopResetEnv(gym.make(env_id), noop_max=30),skip=4)))
        #Applies following wrappers NoopResetEnv(env, noop_max=30) MaxAndSkipEnv(env, skip=4) EpisodicLifeEnv(env) FireResetEnv(env) WarpFrame(env) ClipRewardEnv(env)
        return wrap_deepmind(make_atari(env_id)) 
    else:
        return gym.make(env_id)
def warp(obs):
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
        obs = warp(env.unwrapped._get_obs())
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
    else:
        if isinstance(env.unwrapped,DummyVecEnv):
            return env.unwrapped.envs[0].unwrapped.state
        else:
            return env.unwrapped.state
def de_framestack(obs,*args,**kwargs):
    return obs[...,-1:]
def add_batch_dim(obs,*args,**kwargs):
    return obs[np.newaxis,:]
def warp_obs(obs,env_id,action_prev=None,add_history=True,kill_feats=None,**kw):
    if env_id in ATARI_ENVS:
        if add_history:
            return add_history_to_pixels(obs,action_prev,env_id)
        else:
            return obs
    else:
        if add_history:
            obs = append_history_to_array(obs,action_prev,env_id)
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

def append_history_to_array(obs,action,env_id):
    action = action or 0
    action = 0 if np.isnan(action) else action
    if action not in [0,1,2,1.0,2.0]:
        print(action)
    #print(obs,action)
    if hasattr(action,'len'):
        action = action[0]
    newobs = np.hstack([np.squeeze(obs),action]) #*np.eye(6)[action]
    return newobs

##########   Get experts and trajectories

def get_expert_model(env_id,algo=None):
    '''Require RL Baselines Zoo Package'''
    algo = algo or BEST_ALGO.get(env_id,'dqn')
    general_model_path = ZOO_DIR+algo
    model_path = find_saved_model(algo, general_model_path, env_id, load_best=False)
    #print('Loading {} expert for {}'.format(algo,env_id))
    model = ALGOS[algo].load(model_path, env=None)#make_env(env_id,True))
    return model

def get_trajectories(policy,env_id,N_traj=1,path=None,render=False,verbose=0,df_agg=None,df_E=None,pair_with_E=False,
                     obs_preprocess=None,T_max=None,seed=None,obs_postprocess=None,framestack_env=False,gif_path=None,init_ts=None,expert_after_n=1e10,policy_e=None,e_prepro=None):

    df_columns = ['obs','obs_next','state','state_next','state_prev','action','action_prev','rew','t','traj_ind','weight_action','weight','E_ind']

    #Load trajs_df if exists
    n_traj_loaded = 0
    n_samp_loaded = 0
    if path is not None:
        if os.path.exists(path):
            df_saved = pd.read_pickle(path)
            n_traj_loaded = df_saved['traj_ind'].nunique()
            if n_traj_loaded<N_traj:
                if verbose>0:
                    print('Beginning generation of {} more trajs (found {}).'.format(N_traj-n_loaded,n_loaded))
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
    env = make_env(env_id,framestack_env)
    run_seed = np.random.randint(10000) if seed is None else seed
    env.seed(run_seed)
    episode_rews = []
    start = time.time()
    samp_list = []
    for traj_num in range(n_traj_loaded,N_traj):

        #Handle initialization of initial state and previous action
        env_state, env_state_prev, action_prev, E_inds, t, T_max = None, None, None, None, 0, T_max or 100000
        if pair_with_E and df_E is not None:
            t_init = init_ts[traj_num] if init_ts is not None else 0
            E_inds = df_E[df_E['t']>=t_init][df_E['traj_ind']==traj_num].index
            if len(E_inds)>0:
                T_max = df_E['t'].loc[E_inds].max()
                env_state,env_state_prev,action_prev,t = df_E[['state','state_prev','action_prev','t']].loc[E_inds[0]]
            
        obs = reset_env(env,env_state,env_state_prev,action_prev)
        obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
        obs_deque = deque([np.zeros_like(obs_proc)]*4,maxlen=4)
        episode_rew, done = 0,False
        ep_start_time = time.time()
        while not (done or t>=T_max):
            obs_proc = obs_preprocess(obs,action_prev=action_prev,env_state=env_state,env_id=env_id) if obs_preprocess is not None else obs
            #if verbose==4:
            #    print(obs_proc,obs_proc.shape)
            if t>=expert_after_n:
                obs_deque.append(obs_proc)
                obs_plt = np.stack(obs_deque,-1)[...,-1,:]
                action = policy_e(obs_plt) #handles frame stacking, albeit hard to read
                #print('E',end='')
            else:
                action = policy(obs_proc)
                print(action)
                obs_plt = obs
                #print('L',end='')
            if not framestack_env:
                action = action[0]
            if render:
                env.render(); #time.sleep(.02)
            if t == -1:
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
                axs[0, 0].imshow(obs_plt[0,:,:,0]); axs[0, 1].imshow(obs_plt[0,:,:,1]); axs[1, 0].imshow(obs_plt[0,:,:,2]); axs[1, 1].imshow(obs_plt[0,:,:,3])
                plt.show()
            if gif_path is not None:
                pic = env.unwrapped.envs[0].unwrapped._get_image() if framestack_env else env.unwrapped._get_image()
                img_array.append(pic)
            obs_next, rew, done, _ = env.step(action)
            #print(obs,obs_post(obs))
            env_state_next = get_state(env)
            samp_list.append([obs_post(obs,env_id,action_prev=action_prev),obs_post(obs_next,env_id,action_prev=action),env_state,env_state_next,env_state_prev,action,action_prev,rew,t,traj_num,1.0,1.0,len(samp_list)+n_samp_loaded if E_inds is None else E_inds[t-t_init]])
            env_state,obs,action_prev,env_state_prev = env_state_next,obs_next,action,env_state
            t += 1
            episode_rew += rew
        episode_rews.append(episode_rew)
        if verbose>1:
            print("Episode {}, {} steps ({:.1f} steps/sec) reward: {}".format(traj_num+1,t,t/(time.time()-ep_start_time),episode_rew))
    traj_df = pd.DataFrame(samp_list,columns=df_columns)
    if df_agg is not None:
        traj_df = pd.concat([df_agg,traj_df],ignore_index=True)
    if verbose > 0:
        print("{} episodes, ({:.1f} min, DF {:.1f}MB) Avg reward {:.1f}±{:.1f},".format(N_traj-n_traj_loaded,(time.time()-start)/60,sum(traj_df.memory_usage(deep=True))/2**20,np.mean(episode_rews),np.std(episode_rews)))
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

def keras_NN(Obs_shape,A_dim,H_dims=[8],linear=True,seed=None,n_components=1,cnn=False,NN_mid_as_feats=False):
    '''NN builder func
        Discrete out_dim is A_dim
    '''
    initializer = tf.glorot_uniform_initializer(seed=seed)
    if cnn:
        inp = tf.keras.Input(shape=Obs_shape)
        cast_inp = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32)/tf.cast(255.0,tf.float32))(inp)
        feats = nature_cnn_keras(cast_inp,seed=seed)
        feats2 = tf.keras.layers.Dense(units=32, name='fc2', activation='relu')(feats)
        logits = tf.keras.layers.Dense(units=A_dim, name='pi')(feats2)
        return tf.keras.Model(inputs=inp,outputs=logits)
    if linear:
        model=tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(A_dim,input_shape=Obs_shape,activation=None,kernel_initializer=initializer, bias_initializer=initializer))
        return model
    else:
        model=tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(H_dims[0],input_shape=Obs_shape,activation='tanh',kernel_initializer=initializer, bias_initializer=initializer))
        for i in range(1,len(H_dims)):
            model.add(tf.keras.layers.Dense(H_dims[i],activation='tanh',kernel_initializer=initializer, bias_initializer=initializer))
        model.add(tf.keras.layers.Dense(A_dim,kernel_initializer=initializer))
        return model
def clone_model_and_weights(old_model):
    new_model = tf.keras.models.clone_model(old_model)
    new_model.set_weights(old_model.get_weights())
    return new_model

def train_model(model,dataframe,loss,learning_rate,N_epoch=20,batch_size=32,
                steps_per_epoch=None,verbose=0,seed=None,delta=1e-6,adversary_f=None,df_test=None,test_loss=None,recompute_adversary=None):
    '''trains keras model, either by taking N_epoch*steps_per_epoch optimization
       steps or until step size drops below delta'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataframe['loss'] = np.zeros((len(dataframe),))
    train_loss_results = []
    steps_per_epoch = steps_per_epoch or len(dataframe) #if None, take num_samp steps
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed); np.random.seed(seed)
    last_loss = 1e9
    print_freq = 10000
    start_time = time.time()
    for epoch in range(N_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        random_inds = itertools.cycle(np.random.permutation(dataframe.index.values)) #random shuffle inds
        #epoch_inds = itertools.cycle(np.random.permutation(len(dataframe)//batch_size+1)) #random shuffle inds
        n_steps,print_count = 0,0
        while n_steps<steps_per_epoch:
            batch_indices = [next(random_inds) for i in range(min(batch_size,steps_per_epoch-n_steps))]
            #j = next(epoch_inds)
            #batch_indices = [dataframe.index[i] for i in range(j*batch_size,min((j+1)*batch_size,len(dataframe)))]# + min(steps_per_epoch-n_steps,batch_size))]
            #loss_value, grads = grad(model, dataframe.loc[batch_indices],loss,adversary_f)
            #Compute loss and gradient
            with tf.GradientTape() as tape:
                loss_value = loss(model, dataframe.loc[batch_indices], adversary_f)
                grads = tape.gradient(loss_value, model.trainable_variables)
            #Apply gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            last_loss = np.mean(loss_value)
            dataframe.loc[batch_indices,'loss'] = loss_value.numpy()
            epoch_loss_avg.update_state(loss_value)  # Add list of current losses
            n_steps += len(batch_indices)
        train_loss_results.append(epoch_loss_avg.result())
        if verbose>1:
            tl_str = ' Test Loss: {:.3f}'.format(batch_loss(test_loss,model,df_test)) if df_test is not None else ''
            print("Epoch {:03d}: Train Loss: {:.3f}{}".format(epoch+1,epoch_loss_avg.result(),tl_str))
        if recompute_adversary is not None:
            adversary_f = recompute_adversary(model)
    if verbose>0:
        tl_str = ' Test Loss: {:.3f}'.format(batch_loss(test_loss,model,df_test)) if df_test is not None else ''
        tl_str += ' ({:.1f} min)'.format((time.time()-start_time)/60)
        print("Train Loss ({} Epochs): {:.3f}{}".format(epoch+1,epoch_loss_avg.result(),tl_str))

####### Loss functions

def softmax_cross_entropy(model,dataset,adversary_f=None):
    #Don't apply a softmax at the final layer!! This already does that for you!!
    a = np.hstack(dataset['action'].to_numpy())
    obs = np.vstack(dataset['obs'].to_numpy())
    w = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    #print(a,obs,w)
    #print(tf.compat.v1.losses.sparse_softmax_cross_entropy(a,model(obs),weights=w))
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a,logits=model(obs))*w
    #tf.compat.v1.losses.sparse_softmax_cross_entropy(a,model(obs),weights=w) #This is the same as tf.nn... but with a tf.reduce_mean wrapped around
    #this is equivalent to -np.log(softmax(model(obs),axis=1))[:,a]
def FAIL_loss_old(model,dataset,adversary_f=None):
    obs = np.vstack(dataset['obs'].to_numpy())
    w_E = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    is_expert = tf.cast(dataset['is_expert'].to_numpy(),tf.float32)
    a = tf.cast(np.vstack(dataset['action'].to_numpy()),tf.int32)
    adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next'].to_numpy())),tf.float32))
    action_weight_L = tf.gather_nd(model(obs),a,batch_dims=1)/tf.cast(np.hstack(dataset['action_prob'].to_numpy()),tf.float32)
    #print(action_weight_L.shape,a.shape,model(obs).shape)
    FAIL = action_weight_L*adversary*(1-is_expert) + w_E*adversary*is_expert
    #print(FAIL,adversary,action_weight_L,w,is_expert,(1-is_expert))
    #raise NotImplementedError()
    return FAIL#tf.reduce_mean(FAIL)

def FAIL_loss(model,dataset,adversary_f=None):
    obs = np.vstack(dataset['obs'].to_numpy())
    w_E = tf.cast(dataset['weight'].to_numpy(),tf.float32)
    is_expert = tf.cast(dataset['is_expert'].to_numpy(),tf.float32)
    a = tf.cast(np.vstack(dataset['action'].to_numpy()),tf.int32)
    adversary = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next'].to_numpy())),tf.float32))
    adversary_E = tf.squeeze(tf.cast(adversary_f(np.vstack(dataset['obs_next_orig'].to_numpy())),tf.float32))
    action_weight_L = tf.gather_nd(model(obs),a,batch_dims=1)/tf.cast(np.hstack(dataset['action_prob'].to_numpy()),tf.float32)
    #print(action_weight_L.shape,a.shape,model(obs).shape)
    FAIL = (action_weight_L*adversary-adversary_E)*(1-is_expert) + w_E*adversary*is_expert
    #print(FAIL,adversary,action_weight_L,w,is_expert,(1-is_expert))
    #raise NotImplementedError()
    return FAIL#tf.reduce_mean(FAIL)

def batch_loss_old(loss,model,dataset,adversary_f=None,batch_size=2048):
    return sum([loss(model,dataset[i:i+batch_size],adversary_f).numpy()*len(dataset[i:i+batch_size]) for i in range(0,len(dataset),batch_size)])/len(dataset)
def batch_loss(loss,model,dataset,adversary_f=None,batch_size=2048):
    return np.mean(np.hstack([loss(model,dataset[i:i+batch_size],adversary_f).numpy() for i in range(0,len(dataset),batch_size)]))
def batch_eval(model,dataset,batch_size=2048):
    if len(dataset)<= batch_size:
        return model(dataset)
    return np.vstack([model(dataset[i:i+batch_size]).numpy() for i in range(0,len(dataset),batch_size)])

### Auxiliary

def sample_next_obs(env,state,action,state_prev=None,action_prev=None,n_avg=1):
    '''returns mean next obs over n_avg (sp~P(sp|s,a)) single step simulations'''
    obs_nexts = []
    for i in range(n_avg):
        obs = reset_env(env,state,state_prev,action_prev) #ATARI requires prev state/action for single step roll-in
        obs_nexts.append(env.step(action)[0])
    return np.mean(obs_nexts,axis=0)
def compute_action_prob(df,model,model2prob,batch_size=2048):
    p = np.zeros((len(df),))
    for b in range(0,len(df),batch_size):
        BS = min(len(df)-b,batch_size)
        p[b:b+BS] = model2prob(model(np.vstack(df['obs'][b:b+BS].to_numpy())))[np.arange(BS),np.hstack(df['action'][b:b+BS].to_numpy())]
    return p
def resample_next_states(df_L,sample_env,A_dim,n_samp=1,num_new=None,verbose=0,obs_postprocess = None):
    env_id = sample_env.unwrapped.spec.id
    obs_post = (lambda obs,**kw:obs) if obs_postprocess is None else obs_postprocess
    df_L = df_L.dropna()
    num_new = num_new or len(df_L)
    
    n_samp = min(n_samp,A_dim)
    df = pd.concat([df_L[-num_new:] for n in range(n_samp)],ignore_index=True)
    print('num_new',num_new,len(df))
    df['action_orig'] = df['action']
    df['obs_next_orig'] = df['obs_next']
    if n_samp==A_dim:   #Try every action
        df['action'] = [i for i in range(A_dim) for j in range(len(df)//A_dim)]
    else:               #Randomly subsample actions
        df['action'] = np.random.randint(A_dim,size=(len(df),1))
    #df['action'].plot.hist(bins=6)
    #plt.show()
    df['action_prob'] = np.ones((len(df),1))/A_dim             #Calculate action probability
    start = time.time()
    df['obs_next'] = df.apply(lambda row: obs_post(sample_next_obs(sample_env,row['state'],row['action'],row['state_prev'],row['action_prev']),action_prev=row['action_prev'],env_id=env_id),axis=1)
    df['sp_dist'] = df.apply(lambda row: np.linalg.norm(row['obs_next']-row['obs_next_orig']),axis=1)
    if verbose>0:
        print('Done forward simulating ({:.1f})'.format((time.time()-start)/60))
    df_out = pd.concat([df_L[:-num_new],df],ignore_index=True)
    df_out = df_out.dropna().reset_index(drop=True)
    return df_out

def setup_training_dataframe(alg,df_E,df_L=None,pi_E=None,num_new=None):
    if alg == 'DaD' and df_L is not None:
        df_train = pd.merge(df_L[['obs','t','E_ind']],df_E[['action','weight']].loc[df_L['E_ind']].reset_index(drop=True),left_index=True,right_index=True)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:
        df_E['obs_next_orig'] = df_E['obs_next']
        df_train = pd.concat([df_L,df_E],ignore_index=True).fillna(1)
        df_train['is_expert'] = np.hstack([np.zeros((len(df_L),)),np.ones((len(df_E),))])
        #df_train['loss'] = np.zeros((len(df_train),))
    elif alg in ['DAgger'] and pi_E is not None:
        df_train = df_L.copy()
        df_train['action'] = df_train.apply(lambda row : pi_E(row['action']),axis=1)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-Cov','BC']: #BC, ALICE-Cov
        df_train = df_E
    else:
        raise NotImplementedError('Alg {} training dataframe setup not implemented'.format(alg))
    return df_train

#Keys for ALICE-Co
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
            pipeline_list.append((name+str(ind), None))
        elif name =='poly':
            pipeline_list.append((name+str(ind), PolynomialFeatures(degree=num or 2)))
        elif name =='standardscalar':
            pipeline_list.append((name+str(ind), StandardScaler()))
        elif name =='quantilescalar':
            pipeline_list.append((name+str(ind), QuantileTransformer()))
        elif name =='pca':
            pipeline_list.append((name+str(ind), PCA(n_components=num or 100)))
        elif name =='rff':
            pipeline_list.append((name+str(ind), RBFSampler(n_components=num or 100)))
        else:
            pipeline_list.append((name+str(ind), None))
        #if name == 'signsplit':
        #    pipeline_list.append((name+str(ind), FunctionTransformer(lambda X: np.hstack([X*(X>0),-X*(X<0)]))))
    return pipeline_list
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
            return np.hstack([X_np>0,X_np<0])#np.hstack([X_np*(X_np>0),-X_np*(X_np<0)])
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
    adversary_f = lambda X: np.inner(w,featurize(NN_featurize(X)))
    #print(w)
    #adversary_f = lambda X: np.mean(X,axis=1)
    return adversary_f

def verify_adversary(N=1000,d=5):
    N_samp0,N_samp1 = N//2,N//2
    dim =d
    mu0 = np.zeros((dim,))
    mu1 = np.ones((dim,))*(dim**2)
    sig0 = np.ones((dim,))*2*dim**.25
    sig1 = np.ones((dim,))*dim**.25
    eps = 1e-9
    p_0 = lambda x : np.exp(np.sum(-.5*((mu0-x)/sig0)**2-np.log(sig0)-np.log(2*np.pi)/2,axis=1))
    p_1 = lambda x : np.exp(np.sum(-.5*((mu1-x)/sig1)**2-np.log(sig1)-np.log(2*np.pi)/2,axis=1))
    r_true = lambda x : p_1(x)/(p_0(x)+EPS)+EPS

    x0 = np.random.randn(N_samp0,dim)*sig0 + mu0
    x1 = np.random.randn(N_samp0,dim)*sig1 + mu1
    #y = np.hstack([np.zeros((N_samp0,)),np.ones((N_samp1,))])
    #X = np.vstack([x0,x1])
    adversary_f,w = fit_adversary(x0,x1,feature_map_pipeline='linear')
    plt.scatter(x0[:,0],x0[:,1],marker='x')
    plt.scatter(x1[:,0],x1[:,1],marker='o')
    plt.plot([0,w[0]],[0,w[1]],lw=3,label='w',c='r')
    plt.legend()
    print(w)
    plt.show()

def save_pkl_to_df(path):
    df_columns = ['obs','obs_next','state','state_next','state_prev','action','action_prev','rew','t','traj_ind','weight_action','weight','E_ind']
    trajs = pickle.load(open(path,'rb'))
    samp_list = []
    for traj_num,traj in enumerate(trajs):
        for t,s in enumerate(traj):
            env_state_prev = None if t==0 else samp_list[-1][2]
            action_prev = None if t==0 else samp_list[-1][5]
            samp_list.append([s['o'],s['op'],s['s'],s['sp'],env_state_prev,s['a'],action_prev,s['r'],t,traj_num,1.0,1.0,len(samp_list)])
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

def load_agg_save(path,results_list=[]):
    '''loads df (if exists), appends df in results_list, saves and returns combined df'''
    if os.path.exists(path):
        saved_df = pd.read_csv(path)
        results_list.insert(0,saved_df)
    if len(results_list)==0:
        raise Exception('PEBKAC Error: Nonexistent file {} and no data to add to it'.format(path))
    results_df = pd.concat(results_list,ignore_index=True)
    results_df.to_csv(path,index=False)
    return results_df

def plot_results(df,xaxis,yaxis,lines='algs',filters=None,style='polygon',env_id='',experiment_name=''):
    '''
    Averages accross all columns not specified in constants
    lines - string, list, or dataframe to select which lines to plot
        string or list chooses column(s) from dataframe and plots all unique
            entries/combinations as a separate line.
        dataframe plots each row from dataframe as a separate line
    filters - dict where key is attribute and value is list of permissible
    
    '''
    #Gather Lines
    lines_df = df[lines].drop_duplicates() if type(lines) in [list,str] else lines
    #Filter on constants
    if filters is not None:
        #filters = pd.DataFrame(filters,index=[0])
        df = df[(df[filters.keys()].isin(filters)).all(axis=1)]
    
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lss = ['-','--','-','--','-','--','-','--','-','--','-','--','-','--','-','--']
    xlabel = {'N_E_traj':'Number of Expert Trajectories',
              'N_E_samp':'Number of Expert Samples'}.get(xaxis,xaxis)
    ylabel = {'reward':'On-Policy Reward',
              'loss_train':'Training Loss',
              'loss_test':'Validation Loss',
              'w_ESS':'Effective Sample Size'}.get(yaxis,yaxis)
    plt.figure('-'.join([env_id,experiment_name,ylabel]))
    lines_df = pd.DataFrame(lines_df).reset_index(drop=True)#Handle single line
    for i,line in lines_df.iterrows():
        label = ''.join([str(v) for v in line.values])
        df_line = df[(df[line.keys()]==line.values).all(axis=1)] 
        x = df_line[xaxis].drop_duplicates().to_numpy()
        y = np.array([df_line[df_line[xaxis]==xi][yaxis].mean() for xi in x])
        yerr = np.array([df_line[df_line[xaxis]==xi][yaxis].std() for xi in x])
        if style=='errorbar':
            plt.errorbar(x,y,yerr=yerr,c=cs[i],ls=lss[i],label=label)
        if style=='polygon':
            plt.plot(x,y,c=cs[i],ls=lss[i],label=label,lw=3)
            xy = np.hstack((np.vstack((x,y + yerr)),np.fliplr(np.vstack((x,y - yerr))))).T
            plt.gca().add_patch(Polygon(xy=xy, closed=True, fc=cs[i], lw=2,alpha=.2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(' '.join([env_id,experiment_name,ylabel]))
    plt.legend()

def alg_runner(alg,env_id,**kwargs):
    
    ### Play with these
    N_E_traj = kwargs.setdefault('N_E_traj',10) #Number of expert trajectories to use as test data
    N_agg_iter = kwargs.setdefault('N_agg_iter',10)# if alg!='BC' else 1) #Number of trajectory generation/aggregation iterations
    N_epoch = kwargs.setdefault('N_epoch',5) #Number of training epochs
    total_opt_steps = kwargs.setdefault('total_opt_steps',500000)//N_agg_iter
    add_history = kwargs.setdefault('add_history',False)
    kill_feats = kwargs.setdefault('kill_feats',None)
    verbose = kwargs.setdefault('verbose',0)
    pair_E_s0 = kwargs.setdefault('pair_E_s0',True if alg in ['DaD'] else False)\
    
    run_seed = kwargs.setdefault('run_seed',None)
    opt_seed = kwargs.setdefault('opt_seed',None) #If you are initializing from scratch, this is the weight initializer seed, otherwise this is just batch shuffle seed
    pi0_BC_init = kwargs.setdefault('pi0_BC_init',True) #Loads BC_{N_E_traj} policy as pi0 if present, else, trains one with total_opt_steps
    density_ratio_feature_map = kwargs.setdefault('density_ratio_feature_map','linear')
    adversary_feature_map = kwargs.setdefault('adversary_feature_map','poly 2')

    ### Probably don't play with these
    learning_rate = kwargs.setdefault('learning_rate',0.01)
    H_dims = kwargs.setdefault('H_dims',[64])
    linear = kwargs.setdefault('linear',False)
    N_test_E_traj = kwargs.setdefault('N_test_E_traj',10)
    N_test_rollout = kwargs.setdefault('N_test_rollout',10)
    batch_size = kwargs.setdefault('batch_size',128)
    reinit_opt = kwargs.setdefault('reinit_opt',False)
    resume_BC_training = kwargs.setdefault('resume_BC',False)
    record_intermediate = kwargs.setdefault('record_intermediate',True)
    NN_mid_as_feats = kwargs.setdefault('NN_mid_as_feats',False)
    N_FAIL_samps = kwargs.setdefault('N_FAIL_samps',1)
    render= kwargs.setdefault('render',False)
    switch_2_E_after = kwargs.setdefault('switch_2_E_after',1e10)

    #Save results
    df_results = pd.DataFrame(pd.Series(kwargs,name=0)).T
    results_dicts = [dict() for i in range(10)]
    #Dad_snippets??
    #Expert_noise

    #Get expert, train and test set
    expert_post = de_framestack if env_id in ATARI_ENVS else None
    model_E = get_expert_model(env_id)
    pi_E = lambda obs: model_E.predict(obs)[0] #model_E.predict returns action,state for recurrent policies
    #'cached_experts/'+env_id+'-train.pkl.xz' #'IL_experts/'+env_id+'_demo_trajs.pkl.xz'    #'IL_experts/'+env_id+'_validation_trajs.pkl' 
    df_E = get_trajectories(pi_E,env_id,N_traj=N_E_traj,path='cached_experts/'+env_id+'-train.pkl.xz',T_max=4000,verbose=verbose-2,framestack_env=True,obs_postprocess=expert_post,seed=None)
    df_test = get_trajectories(pi_E,env_id,N_traj=N_test_E_traj,path='cached_experts/'+env_id+'-test.pkl.xz',T_max=4000,verbose=verbose-2,framestack_env=True,obs_postprocess=expert_post,seed=None)
    
    #Add history and kill features on expert 
    learner_pre = lambda *args,**kwargs : add_batch_dim(warp_obs(*args,add_history=add_history,kill_feats=kill_feats,**kwargs))
    Hstr = '{}{}'.format('+H' if add_history else '','-k'+''.join([str(f) for f in kill_feats]) if kill_feats is not None else '')
    df_E[['obs','obs_next']] = df_E.apply(lambda row : (warp_obs(row['obs'],env_id,row['action_prev'],add_history=add_history,kill_feats=kill_feats),
                                                        warp_obs(row['obs_next'],env_id,row['action'],add_history=add_history,kill_feats=kill_feats)),axis=1,result_type='expand')
    df_test[['obs','obs_next']] = df_test.apply(lambda row : (warp_obs(row['obs'],env_id,row['action_prev'],add_history=add_history,kill_feats=kill_feats),
                                                              warp_obs(row['obs_next'],env_id,row['action'],add_history=add_history,kill_feats=kill_feats)),axis=1,result_type='expand')
    if 0:
        save_df_to_pkl(df_E,'IL_experts/'+env_id+'_demo_trajs_H.pkl')
        save_df_to_pkl(df_test,'IL_experts/'+env_id+'_validation_trajs_H.pkl')

    #df_E = df_E[df_E['action_prev']==df_E['action']].reset_index(drop=True)
    #Get NN dims, partial functions, and alg-specific loss func
    sample_env = make_env(env_id) #for sampling next states
    Obs_shape,A_dim = learner_pre(sample_env.reset(),env_id=env_id)[0].shape, sample_env.action_space.n
    test_loss = softmax_cross_entropy
    _keras_NN = partial(keras_NN,Obs_shape=Obs_shape,A_dim=A_dim,cnn=(env_id in ATARI_ENVS),linear=linear,seed=opt_seed,H_dims=H_dims)
    _train_model = partial(train_model,verbose=verbose-1,N_epoch=N_epoch,batch_size=batch_size,steps_per_epoch=int(total_opt_steps/N_agg_iter/N_epoch),seed=opt_seed,learning_rate=learning_rate,df_test=df_test,test_loss=test_loss)
    _get_trajectories = partial(get_trajectories,env_id=env_id,N_traj=N_E_traj,obs_preprocess=learner_pre,obs_postprocess=learner_pre,pair_with_E=pair_E_s0,seed=run_seed,verbose=verbose-1,
expert_after_n=switch_2_E_after,policy_e=pi_E,e_prepro=learner_pre,render=render)
    
    loss = {'BC':softmax_cross_entropy,'DaD':softmax_cross_entropy,'ALICE-Cov':softmax_cross_entropy,'DAgger':softmax_cross_entropy,
            'ALICE-FAIL':FAIL_loss,'ALICE-Cov-FAIL':FAIL_loss,'Expert-FAIL':FAIL_loss}[alg]

    start_time = time.time()

    #Maybe initialize first policy with pre-trained BC, otherwise get busy
    model_list,weights_E_list = [_keras_NN()],[np.ones((len(df_E)))]
    if 0:
        BC_model_path = 'cached_experts/'+env_id+'-BC'+Hstr+'_{}_model_weights.h5'.format(N_E_traj)
        if pi0_BC_init and os.path.exists(BC_model_path): 
            print('Loaded cached BC init params')
            model_list[-1].load_weights(BC_model_path)
            if resume_BC_training:
                _train_model(model_list[-1],df_E,test_loss)
        else:
            _train_model(model_list[-1],df_E,test_loss)
        model_list[-1].save_weights(BC_model_path)
    else:
        _train_model(model_list[-1],df_E,test_loss)

    df_train,adversary_f = df_E,None

    if alg in ['BC','DaD','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:

        df_L = None        
        for i_agg in range(1,N_agg_iter):
            if verbose>1:
                print('{} iter {}, {} train samps'.format(alg+Hstr, i_agg, len(df_train),))
            ### Collect and agg data
            if alg not in ['BC','Expert-FAIL']:
                pi_i = lambda obs: np.argmax(model_list[-1](obs),axis=-1)
                N_L_prev = len(df_L) if df_L is not None else 0
                #print('df_L pre gen',N_L_prev)
                df_L = _get_trajectories(pi_i,df_agg=df_L,df_E=df_E)
                #print('df_L post gen',len(df_L),'num_new',len(df_L)-N_L_prev)
                
            if alg in ['ALICE-FAIL','ALICE-Cov-FAIL']:
                df_L = resample_next_states(df_L,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_L)-N_L_prev,verbose=verbose-1,obs_postprocess=learner_pre)
            if alg in ['Expert-FAIL'] and i_agg==1:
                df_L = resample_next_states(df_E,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_E),verbose=verbose-1,obs_postprocess=learner_pre)
            
            ### Set up training dataset
            #Learn density ratio weighting
            if alg in ['ALICE-Cov','ALICE-Cov-FAIL']:
                weights = estimate_ratio_on_samps(df_L['obs'].values,df_E['obs'].values,density_ratio_feature_map,warm_start=True,verbose=verbose-1)[len(df_L):]
                weights_E_list.append(weights)#/np.mean(weights))
                df_E['weight'] = w_E = np.mean(weights_E_list,axis=0)
                if verbose >1:
                    print('Weights - max {:.1f}, min {:.1f}, effective sample size {:d} ({:d})'.format(max(w_E),min(w_E),int(np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2),len(w_E)))


            #Fit adversary function
            if alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:
                model2prob = lambda outs: softmax(outs,axis=1)
                recompute_adversary = lambda model: fit_adversary(df_L['obs_next'].values,df_E['obs_next'].values,compute_action_prob(df_L,model,model2prob)/df_L['action_prob'].to_numpy(),df_E['weight'].values,adversary_feature_map,NN_mid_as_feats,model,Obs_shape)
                adversary_f = recompute_adversary(model_list[-1])
                
            else:
                adversary_f = None
                recompute_adversary = None
            
            df_train = setup_training_dataframe(alg,df_E,df_L)
            print('df_E',len(df_E),'df_L',len(df_L) if df_L is not None else 0,'df_train',len(df_train))
            ### Train
            new_model = _keras_NN() if reinit_opt else clone_model_and_weights(model_list[-1])
            _train_model(new_model,df_train,loss,adversary_f=adversary_f,recompute_adversary=recompute_adversary)#,N_epoch=1 if i_agg==1 else N_epoch)
            model_list.append(new_model)
            
            ### Record intermediate Results
            
    if alg == 'DAgger':
        raise NotImplementedError()
    df_train.sort_values('E_ind',inplace=True)
    printlist = ['action','loss'] if alg in ['BC'] else ['action_orig','action','loss','sp_dist','E_ind',]
    #print(model_list[-1](df_train['obs'][0]))
    #print(df_train[printlist][:20])
    #print(df_train[printlist][-20:])
    #Choose the best model in the list
    hindsight_losses = [np.mean(batch_loss(loss,model,df_train,adversary_f)) for model in model_list]
    best_ind = np.argmin(hindsight_losses)
    #print(hindsight_losses)
    
    #Score
    #train_loss = loss(model_list[best_ind],df_train,adversary_f)
    test_loss_val = batch_loss(test_loss,model_list[best_ind],df_test,adversary_f)
    test_rollout_df = get_trajectories(lambda obs: np.argmax(model_list[best_ind](obs),axis=-1),env_id,N_traj=N_test_rollout,obs_preprocess=learner_pre,verbose=verbose-2,framestack_env=False,seed=run_seed)
    reward = np.sum(test_rollout_df['rew'].to_numpy())/N_test_rollout
    print('{}   pi_{} train:{:.5f} test:{:.5f} reward:{:.1f} ({:.1f} m)'.format(alg+Hstr,best_ind,hindsight_losses[best_ind],test_loss_val,reward,(time.time()-start_time)/60))
    df_results[['N_E_samp','loss_train','loss_test','reward','runtime','alg','env_id']] = len(df_E),hindsight_losses[best_ind],test_loss_val,reward,(time.time()-start_time)/60,alg+Hstr,env_id
    df_results[['w_max','w_min','w_ESS']] = df_E['weight'].max(),df_E['weight'].min(),np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2
    
    return df_results

if __name__=='__main__':
    env_id = ['CartPole-v1','BeamRiderNoFrameskip-v4','BreakoutNoFrameskip-v4','EnduroNoFrameskip-v4','PongNoFrameskip-v4','QbertNoFrameskip-v4','SeaquestNoFrameskip-v4','SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4','BipedalWalker-v2','LunarLander-v2','LunarLanderContinuous-v2','BipedalWalkerHardcore-v2'][0]
    algo = ['A2C','ACER','ACKTR','PPO2','DQN'][4]
    algo = algo.lower()

    #print('Action space shape',action_dim(env.action_space))

    env_id = ['CartPole-v1','Acrobot-v1','PongNoFrameskip-v4','QbertNoFrameskip-v4','BreakoutNoFrameskip-v4','SpaceInvadersNoFrameskip-v4',][2]
    N_traj = 10
    experiment_name = 'exp0'
    results_path = 'results-'+env_id+'--'+experiment_name+'.csv'
    #options...
    #   Train pong for less time
    #   Train an atari agent for longer
    #   
    if 0:
        model_E = get_expert_model(env_id)#,algo='a2c')
        pi_E = lambda obs: model_E.predict(obs)[0] #None
        traj_df = get_trajectories(pi_E,env_id,N_traj=1,path=None,render=True,verbose=2,gif_path='deepq_expert.gif',T_max=None)
    if 0:
        Obs_shape,A_dim = np.squeeze(traj_df['obs'].loc[0]).shape,make_env(env_id).action_space.n
        L_model_path = env_id+'-BC_{}_model_weights.h5'.format(N_traj)
        L_model = keras_NN(Obs_shape,A_dim,cnn=(env_id in ATARI_ENVS))
        if os.path.exists(L_model_path):
            load_status = L_model.load_weights(L_model_path)
        start = time.time()
        #train_model(L_model,traj_df,BC_loss,0.0001,verbose=1,N_epoch=20,batch_size=128,steps_per_epoch=50000)
        print('Finished in {:.1f} min'.format((time.time()-start)/60))
        #L_model.save_weights(L_model_path)
        env = make_env(env_id)
        obs = env.reset()
        policy_L = lambda obs: np.argmax(L_model(obs),axis=-1)
        print(L_model(obs),policy_L(obs))
        #get_trajectories(policy_L,env_id,N_traj=10,verbose=2,render=True)
    if 0:
        verify_adversary()
    if 0:
        #alg_runner('DaD',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_agg_iter=10,N_E_traj=10)
        #alg_runner('BC',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_E_traj=1)
        df = alg_runner('Expert-FAIL',env_id,verbose=5,total_opt_steps=10000000,N_epoch=4,resume_BC=False,N_E_traj=2,density_ratio_feature_map='poly-1', adversary_feature_map='rff-2048 rff-512',add_history=False,pi0_BC_init=False,N_agg_iter=10,opt_seed=0,run_seed=0,linear=True,kill_feats=None,learning_rate=0.0001,NN_mid_as_feats=True,N_FAIL_samps=3)
        print(df)
    if 1:
        #alg_runner('DaD',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_agg_iter=10,N_E_traj=10)
        #alg_runner('BC',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_E_traj=1)
        df = alg_runner('Expert-FAIL','Acrobot-v1',verbose=5,total_opt_steps=1000000,N_epoch=4,resume_BC=False,N_E_traj=2,density_ratio_feature_map='poly-1', adversary_feature_map='rff-512',add_history=False,pi0_BC_init=False,N_agg_iter=5,opt_seed=0,run_seed=0,linear=True,kill_feats=None,learning_rate=0.1,NN_mid_as_feats=False,N_FAIL_samps=3)
        print(df)
    if 0:
        save_pkl_to_df('IL_experts/'+env_id+'_demo_trajs.pkl')    
        save_pkl_to_df('IL_experts/'+env_id+'_validation_trajs.pkl')

    if 0:

        results_list = []
        for i in range(10):
          for j in range(2):
            results_list.append(alg_runner('BC',env_id,verbose=0,total_opt_steps=600000,N_epoch=3,resume_BC=False,N_E_traj=i+1,density_ratio_feature_map='poly-1', adversary_feature_map='poly-2',add_history=True,pi0_BC_init=False,N_agg_iter=5,opt_seed=None,run_seed=None,linear=True,kill_feats=None))
            #results_list.append(alg_runner('BC',env_id,verbose=0,total_opt_steps=600000,N_epoch=3,resume_BC=False,N_E_traj=i+1,density_ratio_feature_map='poly-1', adversary_feature_map='poly-2',add_history=False,pi0_BC_init=False,N_agg_iter=5,opt_seed=None,run_seed=None,linear=True,kill_feats=[5]))
        results_df = load_agg_save(results_path,results_list)
        #results_df = pd.concat(results_list,ignore_index=True)
        plot_results(results_df,'N_E_traj','reward','alg',filters=None,env_id =env_id,experiment_name=experiment_name)
        plot_results(results_df,'N_E_traj','loss_train',['alg'],filters=None,env_id=env_id,experiment_name=experiment_name)
        plot_results(results_df,'N_E_traj','loss_test',['alg'],filters=None,env_id=env_id,experiment_name=experiment_name)
        plt.show()
    if 0:
        #Test adding history
        env = make_env(env_id)
        obs = env.reset()
        for i in range(200):
            obs = env.step([np.random.randint(env.action_space.n)])[0]
        obs = add_history_to_pixels(obs,0,env_id)
        plt.imshow(np.squeeze(de_framestack(obs)))
        plt.show()
    if 0:
        #df_E = get_trajectories(None,env_id,N_traj=N_traj,path='cached_experts/'+env_id+'-train.pkl.xz',T_max=4000)
        #obs = df_E['obs'][400]
        #plt.imshow(obs[0,:,:,0])
        #plt.show()
        env = make_env(env_id)
        env.reset()
        for i in range(500):
            action = np.random.randint(6)
            obs = env.step(action)[0]
        plt.imshow(np.squeeze(obs))
        plt.show()
    if 0:
        #Test reset
        df_E = get_trajectories(None,env_id,N_traj=N_traj,path='cached_experts/'+env_id+'-train.pkl.xz',T_max=4000)
        Obs_shape,A_dim = df_E['obs'].loc[0][0].shape,make_env(env_id).action_space.n
        #make_session() #
        #model_E = get_expert_model(env_id)
        #model_L = keras_NN(Obs_shape,A_dim,cnn=True)
        #model_L.load_weights('cached_experts/PongNoFrameskip-v4-BC_10_model_weights_toogood.h5')
        #pi_L = lambda obs: np.argmax(model_L(obs),axis=-1)
        env = make_env(env_id)
        obs = reset_env(env)
        action_prev,env_state = None,None
        for i in range(200):
            #obs = env.step(pi_L(add_batch_dim(obs)))[0]
            action = np.random.randint(6)
            obs_next, rew, done, _ = env.step(action)
            env_state_next = get_state(env)
            #Save point
            if i == 198:
                break
            env_state,obs,action_prev,env_state_prev = env_state_next,obs_next,action,env_state

        plt.imshow(np.squeeze(obs))
        obs_r = reset_env(env)
        env.step(0)
        for i in range(10):
            env.step(0)
        obs_r = env.step(0)[0]
        
        plt.figure(); plt.imshow(np.squeeze(obs_r))
        obs_rs = reset_env(env,None,env_state_prev,action_prev)
        plt.figure(); plt.imshow(np.squeeze(obs_rs))
        plt.figure(); plt.imshow(np.squeeze(obs_rs-obs))

        plt.show()

    if 0:
        results_file

    #Pong
    #N_E    #training     train_loss    test loss  reward  time_elapsed
    #10     1000000         .364         .405       -10.6       

    #ALICE Acrobot was a bit of a fluke. 
    #2 issues
    #BC v BC+H try out DhHann's doesn't actually create issues
    # Look in the literature for all the examples where +H supposedly create issues
    # Try to actually tear down every place where the claim that BC or BC+H doesn't work
        #Need to augment the set of reproducible problems



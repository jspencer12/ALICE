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
import gym
import time, os, itertools, sys, pickle, yaml, subprocess, copy, portalocker

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv, DummyVecEnv
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
def reshape_recolor_atari(obs):
     return cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)[:,:,None]

def add_batch_dim(obs,*args,**kwargs):
    return obs[np.newaxis,:]

##########   Get experts and trajectories
def train_zoo_agent(env_id,RL_algo='dqn',RL_expert_folder='my_RL_experts',n_steps=None,env_kwargs=None,hyperparams=None):
    ''' Train an RL agent using the rl_baselines_zoo package (must be installed in same directory)
    
        se 
    
        imputs:
            env_id           - str, name of environment (eg. "CartPole-v1")
            RL_algo          - str, one of: 'dqn','a2c','trpo','ppo2','sac','acktr'
            RL_expert_folder - str, directory to save policy
            n_steps          - int, total number of environment training steps
            
        '''
    #assert os.path.exists('rl_baselines_zoo')
    start = time.time()
    #parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    #parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    #parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    #parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    #parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',default=10000, type=int)
    #parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',default=5, type=int)
    #parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',default=-1, type=int)
    #parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    #parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    #parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,type=int)
    #parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    #parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,help='Optional keyword argument to pass to the env constructor')
    arglist = ['python',THIS_DIR+'/rl_baselines_zoo/train.py','--algo',RL_algo,
               '--env',env_id,'-n',str(n_steps),'-f',RL_expert_folder]
    print(arglist)
    subprocess.run(arglist)
    print('Finished training after {:.1f} minutes'.format((time.time()-start)/60))


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
    
def get_trajectories(policy,env_id,N_traj=1,render=False,random_seed=None,verbose=0,path=None):
    ''' Generate new trajectories or load previously saved trajectories
    
        inputs:
            policy - fu
    '''

    #Load trajs_df if path exists
    N_loaded = 0
    df_saved = pd.DataFrame()
    if path is not None:
        if os.path.exists(path):
            df_saved = pd.read_pickle(path)
            N_loaded = df_saved['traj_ind'].nunique()
            if N_loaded >= N_traj:
                if verbose>0:
                    print('Loaded {} trajs from {}'.format(N_traj,path))
                return df_saved[df_saved['traj_ind']<N_traj]

    env = make_env(env_id,vec_env)
    np.random.seed()
    random_seed = np.random.randint(10000) if run_seed is None else run_seed
    env.seed(run_seed)   #Some environments default to seed=0 if you don't specify
    episode_rews = []
    start = time.time()
    trajs = []
    for traj_ind in range(n_traj_loaded,N_traj):
        ep_start_time = time.time()
        episode_rews.append(0)
        t, done = 0, False
        obs = env.reset()
        while not done:
            action = policy(obs)
            if render:
                env.render(); time.sleep(.01)
            obs_next, rew, done, _ = env.step(action)
            rew = rew[0] if hasattr(rew,'len') else rew    #make sure reward is scalar
            trajs.append({'obs':obs,'action',action,'obs_next':obs_next,'rew':rew,'t':t,'traj_ind':traj_ind})
            episode_rews[-1] += rew
            t += 1
        if verbose>1:
            print("Episode {}, {} steps ({:.1f} steps/sec) reward: {}".format(traj_ind,t,t/(time.time()-ep_start_time),episode_rews[-1]))
    if verbose > 0:
        print("Generated {} trajs, ({:.1f} min) Avg reward {:.1f}+-{:.1f},".format(N_traj-N_loaded,(time.time()-start)/60,np.mean(episode_rews),np.std(episode_rews)))
    env.close()
    traj_df = pd.concat([df_saved,pd.DataFrame(trajs)],ignore_index=True)
    if path is not None:
        if os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        traj_df.to_pickle(path)
        if verbose>0:
            print('Saved {} trajs to {}'.format(N_traj,path))
    return traj_df

##########    Build Network and Train

def keras_NN(Obs_shape,A_dim,H_dims=(8,),linear=True,cnn=False,random_seed=None,activation='relu'):
    ''' Neural Network Builder/Initializer Function
    
        There is no regularization at the output layer, so for discrete action spaces, this is
        the output are the logit inputs to be fed into a softmax function for generating action
        probabilities, or for a continuous action space, the outputs are directly actions.
        
        inputs:
            Obs_shape   - tuple, dimensions of observation
            A_dim       - int, action dimension
            H_dims      - tuple, dimensions of each hidden layer
            linear      - bool, if True, the network is a simple linear network with no regularization
            cnn         - bool, for image inputs. Uses Nature CNN architecture to preprocess obs and produce 512 features
            random_seed - int, seed for initializing weights/biases
            activation  - str, activation func after every layer (eg. 'relu', 'tanh', 'sigmoid', None)
        output:
            model      -  keras/tf trainable modeol
    '''
    #Deal with H_dims/activation for linear NN
    activation = None if linear else activation
    H_dims = () if linear else H_dims
    
    layer_output_dims = [*H_dims,A_dim]
    init = tf.glorot_uniform_initializer(seed=random_seed) # Random seed for initializer
    inp = tf.keras.input(shape=Obs_shape)
    
    #Featurize inputs or flatten them
    if cnn:
        cast_inp = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32)/tf.cast(255.0,tf.float32))(inp)
        ### CNN architecture from the nature paper, input tensor (image) returns 1D tensor (features)
        layer_1 = tf.keras.layers.Conv2D(filters=32, name='c1', kernel_size=8, strides=4, activation='relu',kernel_initializer=init)(inputs)
        layer_2 = tf.keras.layers.Conv2D(filters=64, name='c2', kernel_size=4, strides=2, activation='relu',kernel_initializer=init)(layer_1)
        layer_3 = tf.keras.layers.Conv2D(filters=64, name='c3', kernel_size=3, strides=1, activation='relu',kernel_initializer=init)(layer_2)
        layer_3 = tf.keras.Flatten()(layer_3)
        out = tf.keras.layers.Dense(units=512, name='fc1', activation='relu')(layer_3)
    else:
        out = tf.keras.layers.Flatten()(inp)
    
    #Add hidden/output layers
    for units in layer_output_dims:
        out = tf.keras.layers.Dense(units=units, activation=activation, kernel_initializer=init, bias_initializer=init)(out)

    return tf.keras.Model(inputs=inp,outputs=out)

def clone_model_and_weights(old_model):
    new_model = tf.keras.models.clone_model(old_model)
    new_model.set_weights(old_model.get_weights())
    return new_model

def train_model(model,df_train,train_loss,learning_rate,N_epoch=20,batch_size=32,steps_per_epoch=None,verbose=0,random_seed=None,df_test=None,test_loss=None):
    ''' Trains keras model using samples in df_train
        
        inputs:
            model           - tf/keras model
            df_train        - pandas dataframe of training data with columns 'action','obs' and maybe 'weight'
            train_loss      - function(model,dataframe)->returns one loss per item in dataframe
            learning_rate   - float, Adam learning rate
            N_epoch         - int, number of training epochs
            batch_size      - samples per gradient update step
            steps_per_epoch - int, [len(df_train)] number of steps to take in an epoch
            verbose         - int, 0-Nothing, 1-Print summary at end, 2-Print after every epoch
            random_seed     - int, random seed for shuffling indices in creating batches
            df_test         - pandas dataframe of training data with columns 'action','obs' and maybe 'weight'
            
    , either by taking N_epoch*steps_per_epoch optimization
       steps or until step size drops below delta'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    steps_per_epoch = steps_per_epoch or len(df_train) #if None, take num_samp steps
    np.random.seed(randm_seed)
    tf.compat.v1.set_random_seed(random_seed)
    print_freq = 10000
    start_time = time.time()
    for epoch in range(N_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        random_inds = itertools.cycle(np.random.permutation(df_train.index.values)) #random shuffle inds
        n_steps = 0
        while n_steps<steps_per_epoch:
            #grab a batch of samples until you hit desired steps_per_epoch
            batch_inds = [next(random_inds) for i in range(min(batch_size,steps_per_epoch-n_steps))]
            with tf.GradientTape() as tape:
                #compute loss/gradient
                loss_value = train_loss(model, df_train.loc[batch_inds], adversary_f,model_prob_a,entropy_coeff=entropy_coeff)
                grads = tape.gradient(loss_value, model.trainable_variables)
            #Apply gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            last_loss = np.mean(loss_value)
            epoch_loss_avg.update_state(loss_value)  # Add list of current losses
            n_steps += len(batch_indices)
        # Epoch done - print performance
        if verbose>1:
            tl_str = ' Test Loss: {:.3f}'.format(batch_loss(test_loss,model,df_test)) if df_test is not None else ''
            print("Epoch {:03d}: Train Loss: {:.3f}{}".format(epoch+1,epoch_loss_avg.result(),tl_str))
    # Training done - print performance
    if verbose>0:
        tl_str = ' Test Loss: {:.3f}'.format(batch_loss(test_loss,model,df_test)) if df_test is not None else ''
        tl_str += ' ({:.1f} min)'.format((time.time()-start_time)/60)
        print("Train Loss ({} Epochs): {:.3f}{}".format(epoch+1,epoch_loss_avg.result(),tl_str))
    #if 'is_expert' in df_train.columns:
 #       df_train[df_train['is_expert']==False].hist(column='loss', bins=100)
  #  else:
   #     df_train.hist(column='loss', bins=100)
    #plt.show()
    return train_results

##### Loss functions

def softmax_cross_entropy(model,df):
    #Don't apply a softmax at the final layer!! This loss already does that for you!!
    a = np.hstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a,logits=model(obs))*w #this is equivalent to -np.log(softmax(model(obs),axis=1))[:,a]

def zeroone_loss(model,df):
    a = np.hstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.cast(tf.math.not_equal(a,tf.math.argmax(model(obs),axis=1)),tf.float32)*w
    
def L2_loss(model,df):
    a = np.vstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.sqrt(tf.reduce_sum((model(obs)-a)**2,axis=1))*w
    
def mse_loss(model,df):
    a = np.vstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.reduce_sum((model(obs)-a)**2,axis=1)*w
    
def L1_loss(model,df):
    a = np.vstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.reduce_sum(tf.abs(model(obs)-a),axis=1)*w
    
def logcosh_loss(model,df):
    a = np.vstack(df['action'].to_numpy())
    obs = np.vstack(df['obs'].to_numpy())
    w = tf.cast(df['weight'].to_numpy(),tf.float32) if 'weight' in df.columns else 1
    return tf.reduce_sum(tf.log(tf.cosh(model(obs)-a)),axis=1)*w

def batch_loss(loss,model,df,batch_size=2048):
    # reduce cpu/gpu load by evaluating loss in batch
    return np.mean(np.hstack([loss(model,df[i:i+batch_size]).numpy() for i in range(0,len(df),batch_size)]))
    
def batch_eval(model,df,batch_size=2048):
    # reduce cpu/gpu load by evaluating model in batch
    if len(df)<= batch_size:
        return model(df)
    return np.vstack([model(df[i:i+batch_size]).numpy() for i in range(0,len(df),batch_size)])

### Auxiliary

def setup_training_df_train(alg,df_E,df_L=None,pi_E=None,num_new=None):
    if alg == 'DaD' and df_L is not None:
        df_train = pd.merge(df_L[['obs','t','E_ind']],df_E[['action','weight']].loc[df_L['E_ind']].reset_index(drop=True),left_index=True,right_index=True)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:
        df_E['obs_next_ref'] = df_E['obs_next']
        df_E['obs_next_E'] = df_E['obs_next']
        df_E['action_ref'] = df_E['action']
        df_L = pd.merge(df_L,df_E[['obs_next_E']].loc[df_L['E_ind']].reset_index(drop=True),left_index=True,right_index=True)
        df_train = pd.concat([df_L,df_E],ignore_index=True).fillna(1)
        df_train['is_expert'] = np.hstack([np.zeros((len(df_L),)),np.ones((len(df_E),))])
        #df_train['loss'] = np.zeros((len(df_train),))
    elif alg in ['DAgger'] and pi_E is not None:
        df_train = df_L.copy()
        df_train['action'] = df_train.apply(lambda row : pi_E(row['obs']),axis=1)
        df_train = pd.concat([df_train,df_E],ignore_index=True)
    elif alg in ['ALICE-Cov','BC']: #BC, ALICE-Cov
        df_train = df_E
    else:
        raise NotImplementedError('Alg {} training df_train setup not implemented'.format(alg))
    return df_train

def load_agg_save(path,results_list=[]):
    '''loads df (if exists), appends df in results_list, saves and returns combined df'''
    results_list = [results_list] if type(results_list) is not list else results_list #idiot proof
    if len(results_list)>0:
        with portalocker.Lock(path) as f:
            results_df = pd.concat(results_list,ignore_index=True)
            results_df.to_csv(f,index=False,header=f.tell()==0) #Adds header only to first line
    return pd.read_csv(path)

def plot_results(df,xaxis,yaxis,lines='alg',filters=None,**plotattrs):
    '''
    Averages accross all columns not specified in constants
    lines - string, list, or df_train to select which lines to plot
        string or list chooses column(s) from df_train and plots all unique
            entries/combinations as a separate line.
        df_train plots each row from df_train as a separate line
    filters - dict where key is attribute and value is list of permissible
    
    '''
    if type(df) is str:
        df = load_agg_save(df)
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
                             'recent_samp_priority_exp':'pe'}.get(col,col)

    title = plotattrs.setdefault('title',' '.join([env_id,exp_name,ylabel]))

    #Add lines
    figname = '-'.join([env_id,exp_name,xaxis,yaxis])
    plt.figure(figname)
    lines_df = pd.DataFrame(lines_df).reset_index(drop=True) #Handle single line
    for i,line in lines_df.iterrows():
        label = ', '.join([leg_label(k)+' = '+leg_format(k).format(v) if k!='alg' else v for k,v in line.items()])
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
        plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir,figname))


def alg_runner(alg,env_id,**kwargs):
    
    #useful booleans
    is_atari = env_id in ATARI_ENVS
    is_cov =  alg in ['ALICE-Cov','ALICE-Cov-FAIL']
    is_fail = alg in ['ALICE-FAIL','ALICE-Cov-FAIL']
    is_alice = is_cov or is_fail
    is_mujoco = env_id in MUJOCO_ENVS

    ### Play with these
    N_E_traj = kwargs.setdefault('N_E_traj',10) #Number of expert trajectories to use as test data
    N_ALICE_traj = kwargs.setdefault('N_ALICE_traj',N_E_traj)
    N_agg_iter = kwargs.setdefault('N_agg_iter',10 if alg not in ['Expert','Random'] else 1)
    N_epoch = kwargs.setdefault('N_epoch',5) #Number of training epochs
    opt_steps_per_iter = kwargs.setdefault('total_opt_steps',500000)//N_agg_iter
    add_history = kwargs.setdefault('add_history',False)
    kill_feats = kwargs.setdefault('kill_feats',None)
    if kill_feats is not None:
        kill_feats = kwargs['kill_feats'] = tuple(kill_feats) if type(kill_feats) in [list,tuple] else (kill_feats,)
    verbose = kwargs.setdefault('verbose',0)
    
    run_seed = kwargs.setdefault('run_seed',None)
    opt_seed = kwargs.setdefault('opt_seed',None) #If you are initializing from scratch, this is the weight initializer seed, otherwise this is just batch shuffle seed
    
    density_ratio_feature_map = kwargs.setdefault('density_ratio_feature_map','linear' if is_cov else None)
    adversary_feature_map = kwargs.setdefault('adversary_feature_map','poly 2' if is_fail else None)

    ### Probably don't play with these
    learning_rate = kwargs.setdefault('learning_rate',0.01)
    learning_rate_BC = kwargs.setdefault('learning_rate_BC',learning_rate)
    entropy_coeff = kwargs.setdefault('entropy_coeff',None)                     
    H_dims = kwargs.setdefault('H_dims',(512,) if not is_atari else None)
    if H_dims is not None:
        H_dims = kwargs['H_dims'] = tuple(H_dims) if type(H_dims) in [list,tuple] else (H_dims,)
    linear = kwargs.setdefault('linear',False)
    N_test_E_traj = kwargs.setdefault('N_test_E_traj',50)                       #Test df size
    N_test_rollout = kwargs.setdefault('N_test_rollout',50)                     #N trajs for final rollout
    batch_size = kwargs.setdefault('batch_size',128)                            #Optimization batch sizes
    reinit_opt = kwargs.setdefault('reinit_opt',False)                          #Reinitialize optimization at each agg iter or build on previous parameters
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
    T_max = kwargs.setdefault('T_max',None)                                     #Max timesteps in eval environments
    model_reg_coeff = kwargs.setdefault('model_reg_coeff',None)                 #amount of regularization to apply to each of 
    partial_obs = kwargs.setdefault('partial_obs',False)
    results_path = kwargs.setdefault('results_path',None)
    recent_samp_priority_exp = kwargs.setdefault('recent_samp_priority_exp',1)
    horizon_weight_offset_exp = kwargs.setdefault('horizon_weight_offset_exp',None)
    pair_with_E = kwargs.setdefault('pair_with_E',True if (alg in ['DaD']) or (horizon_weight_offset_exp is not None) else False)
    drop_first = kwargs.setdefault('drop_first',0)

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
    print('Train set reward',np.sum(df_E['rew'].to_numpy())/N_E_traj,'Test set reward',np.sum(df_test['rew'].to_numpy())/N_test_E_traj)

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
        model_prob_a = lambda model_out,a: tf.gather_nd(tf.nn.softmax(model_out,axis=-1),a,batch_dims=1)
        clip_range = None
    else:
        model_prob_a = lambda model_out,a: 0
        clip_range = [sample_env.action_space.low,sample_env.action_space.high]

    #df_E = df_E[df_E['action_prev']==df_E['action']].reset_index(drop=True)
    #Get NN dims, partial functions, and alg-specific loss func

    Obs_shape = df_E['obs'][0].shape[1:] if (alg=='Expert' or is_atari) else add_batch_dim(df_E['obs'][0]).shape[1:]
    _keras_NN = partial(keras_NN,Obs_shape=Obs_shape,A_dim=A_dim,cnn=is_atari,linear=linear,seed=opt_seed,H_dims=H_dims,model_reg_coeff=model_reg_coeff,clip_range=clip_range)
    _train_model = partial(train_model,verbose=verbose-1,N_epoch=N_epoch,batch_size=batch_size,steps_per_epoch=int(opt_steps_per_iter/N_epoch),seed=opt_seed,learning_rate=learning_rate,df_test=df_test,test_loss=test_loss,model_prob_a=model_prob_a,entropy_coeff=entropy_coeff)
    _get_trajectories = partial(get_trajectories,env_id=env_id,N_traj=N_ALICE_traj,obs_preprocess=learner_pre,obs_postprocess=learner_pre,pair_with_E=pair_with_E,
                                verbose=verbose-1,expert_after_n=switch_2_E_after,policy_e=pi_E,e_prepro=learner_pre,render=render,T_max=T_max,
                                randinit_t=horizon_weight_offset_exp is not None,choose_random_expert=(N_ALICE_traj!=N_E_traj))
    _eval_pi_lightweight = partial(eval_pi_lightweight,env_id=env_id,N_traj=N_test_rollout,run_seed=run_seed,obs_preprocess=learner_pre,verbose=verbose-1)


    start_time = time.time()

    #Initialize model list
    kwargs.update(alg = alg + Hstr, env_id = env_id, N_E_samp = len(df_E))
    results_dicts = [copy.copy(kwargs) for i in range(N_agg_iter)]
    model_list,weights_E_list = [_keras_NN()],[np.ones((len(df_E)))]
    df_train,adversary_f,df_L = df_E,None,None
    epoch_train_losses = []
    df_E['i_agg'] = np.zeros(len(df_E))

    if alg in ['BC','DaD','ALICE-Cov','ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:

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
                pi_i = lambda obs: np.squeeze(np.argmax(model_list[-1](obs),axis=-1)) if DISCRETE else np.squeeze(model_list[-1](obs))
                N_L_prev = len(df_L) if df_L is not None else 0
                print('df_L pre gen',N_L_prev)
                horizon = max_len if horizon_weight_offset_exp is None else min(int(horizon_weight_offset_exp[0]*(i_agg**horizon_weight_offset_exp[2]) + horizon_weight_offset_exp[1]),max_len)
                df_L = _get_trajectories(pi_i,df_agg=df_L,df_E=df_E,horizon=horizon,seed=run_seed+1000*i_agg)
                df_L.loc[np.arange(N_L_prev,len(df_L)),'i_agg'] = np.ones(len(df_L)-N_L_prev)*i_agg
                
                df_L = df_L[df_L['t']>=drop_first]
                
                if is_fail:
                    df_L.dropna()
                
                print('df_L post gen',len(df_L),'num_new',len(df_L)-N_L_prev,'horizon',horizon)

            ### Save intermediate results
            if horizon_weight_offset_exp is not None:
                reward,reward_std = _eval_pi_lightweight(pi_i,N_traj=10)
            else:
                rewards = [np.sum(df_L[N_L_prev:][df_L['traj_ind']==i]['rew'].to_numpy()) for i in pd.unique(df_L[N_L_prev:]['traj_ind'])]
                print(rewards)
                reward,reward_std = np.mean(rewards),np.std(rewards)
            print('rew/std',reward,reward_std)
            #hindsight_losses = [batch_loss(train_loss,model,df_train,adversary_f) for model in model_list]
            #best_ind = np.argmin(hindsight_losses) if alg != 'BC' else -1
            results_dicts[i_agg-1].update({'w_max':df_E['weight'].max(), 'w_min':df_E['weight'].min(), 'final':False, 'iteration_num':i_agg-1,'horizon':horizon,
                                          'w_ESS':np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2,
                                          'reward':reward,'reward_std':reward_std, 'total_opt_steps':opt_steps_per_iter*i_agg,
                                          'loss_test':batch_loss(test_loss,model_list[-1],df_test,adversary_f,model_prob_a),
                                          'loss_train':batch_loss(train_loss if i_agg>1 else test_loss,model_list[-1],df_train,adversary_f,model_prob_a,entropy_coeff),
                                          'class_test':batch_loss(classification_loss,model_list[-1],df_test,adversary_f,model_prob_a),
                                          'runtime':(time.time()-start_time)/60,'best_ind':i_agg-1,'JS_div':js_from_samples(df_L['obs_next'].values,df_E['obs_next'].values)})
            #print(df_L['i_agg'].values)
            if is_fail:
                df_L = resample_next_states(df_L,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_L)-N_L_prev,verbose=verbose-1,obs_postprocess=learner_pre)
            if alg in ['Expert-FAIL'] and i_agg==1:
                df_L = resample_next_states(df_E,sample_env,A_dim,n_samp=N_FAIL_samps,num_new=len(df_E),verbose=verbose-1,obs_postprocess=learner_pre)
            
            ### Set up training df
            #Learn density ratio weighting
            if is_cov:
                weights = estimate_ratio_on_samps(df_L['obs'].values,df_E['obs'].values,density_ratio_feature_map,warm_start=True,verbose=verbose-1)[len(df_L):]
                weights_E_list.append(weights)#/np.mean(weights))
                df_E['weight'] = w_E = np.mean(weights_E_list,axis=0)
                if verbose >1:
                    print('Weights - max {:.1f}, min {:.1f}, effective sample size {:d} ({:d})'.format(max(w_E),min(w_E),int(np.linalg.norm(w_E,ord=1)**2/np.linalg.norm(w_E,ord=2)**2),len(w_E)))
            #Reweigh learner
            if is_fail:
                #print(df_L['i_agg'].values,np.power(recent_samp_priority_exp,i_agg-df_L['i_agg'].values))
                df_L['weight'] = np.power(recent_samp_priority_exp,i_agg-df_L['i_agg'].values)
                if verbose>1:
                    print('min_weight',df_L['weight'].min(),'max_weight',df_L['weight'].max())

            #Fit adversary function
            if alg in ['ALICE-FAIL','ALICE-Cov-FAIL','Expert-FAIL']:
                recompute_adversary = lambda model: fit_adversary(df_L['obs_next_ref'].values,df_E['obs_next'].values,df_L['weight'].values*compute_action_prob(df_L,model,model_prob_a)/df_L['action_ref_prob'].to_numpy(),df_E['weight'].values,adversary_feature_map,NN_mid_as_feats,model,Obs_shape)
                adversary_f = recompute_adversary(model_list[-1])
                
            else:
                adversary_f = None
                recompute_adversary = None
            
            df_train = setup_training_df_train(alg,df_E,df_L)
            #print('df_E',len(df_E),'df_L',len(df_L) if df_L is not None else 0,'df_train',len(df_train))
            
            ### Train
            new_model = _keras_NN() if reinit_opt else clone_model_and_weights(model_list[-1])
            train_results = _train_model(new_model,df_train,train_loss,adversary_f=adversary_f,recompute_adversary=recompute_adversary)
            results_dicts[i_agg].update(train_results)
            epoch_train_losses.extend(train_results['epoch_losses'])
            model_list.append(new_model)
            #print(df_train['loss'].min(),df_train['loss'].max(),df_train['loss'].mean())

    if alg == 'Expert':
        model_list[0] = lambda obs: model_E.proba_step(np.reshape(obs,(-1,*Obs_shape))) if DISCRETE else model_E.predict(np.reshape(obs,(-1,*Obs_shape)))

    if alg == 'Random':
        model_list[0] = lambda obs: np.random.rand(obs.shape[0],A_dim).astype(np.single)

    if alg == 'DAgger':
        raise NotImplementedError()

    if 0 and (verbose>0):
        print('Epoch Losses: '+' '.join([f'{loss:.2g}' for loss in epoch_train_losses]))
    #df_train.sort_values('E_ind',inplace=True)
    #printlist = ['action','loss'] if alg in ['BC'] else ['action_orig','action','loss','sp_dist','E_ind',]
    #print(model_list[-1](df_train['obs'][0]))
    #print(df_train[printlist][:20])
    #print(df_train[printlist][-20:])

    #Choose the best model in the list
    hindsight_losses = [batch_loss(train_loss,model,df_train,adversary_f,model_prob_a,entropy_coeff) for model in model_list]
    for i in range(len(results_dicts)):
        results_dicts[i]['hindsight_loss_train'] = hindsight_losses[i]
    best_ind = np.argmin(hindsight_losses) if alg != 'BC' else len(model_list)-1
    print('Hindsight Losses:',', '.join([f'{L:.2f}' for L in hindsight_losses]))
    
    #Score
    test_loss_val = batch_loss(test_loss,model_list[best_ind],df_test,adversary_f,model_prob_a)
    pi_rollout = lambda obs: np.squeeze(np.argmax(model_list[best_ind](obs),axis=-1)) if DISCRETE else np.squeeze(model_list[best_ind](obs))
    #for i,row in df_train[['obs','action']][:10].iterrows():
    #    print(row['action'],model_list[-1](row['obs']),pi_rollout(row['obs']))
    test_rollout_df = get_trajectories(pi_rollout,env_id,N_traj=N_test_rollout,render=render_final,
                                       obs_preprocess=learner_pre,verbose=verbose-2,vec_env=((alg=='Expert') and is_atari),seed=run_seed,T_max=T_max)
    rewards = [np.sum(test_rollout_df[test_rollout_df['traj_ind']==i]['rew'].to_numpy()) for i in range(N_test_rollout)]
    reward,reward_std = np.mean(rewards),np.std(rewards)
    print('{} {} {}   pi_{} train:{:.5f} test:{:.5f} reward:{:.1f} ({:.1f} m)'.format(env_id,N_E_traj,alg+Hstr,best_ind,hindsight_losses[best_ind],test_loss_val,reward,(time.time()-start_time)/60))

    results_dicts[-1].update({'w_max':df_E['weight'].max(), 'w_min':df_E['weight'].min(), 'final':True, 'iteration_num':i_agg,'horizon':max_len,
                              'w_ESS':np.linalg.norm(df_E['weight'].to_numpy(),ord=1)**2/np.linalg.norm(df_E['weight'].to_numpy(),ord=2)**2,
                              'reward':reward, 'reward_std':reward_std,'loss_test':test_loss_val,
                              'loss_train':hindsight_losses[best_ind],'JS_div':js_from_samples(df_L['obs_next'].values,df_E['obs_next'].values),
                              'class_test':batch_loss(classification_loss,model_list[best_ind],df_test,adversary_f,model_prob_a),
                              'runtime':(time.time()-start_time)/60,'best_ind':best_ind})
    sample_env.close()

    results_df = pd.DataFrame(results_dicts)
    if results_path is not None:
        load_agg_save(results_path,results_df)
    return results_df
    
if __name__=='__main__':
    pass
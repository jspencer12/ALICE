import ALICE, os, subprocess
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) #hacky to get tf quiet
warnings.filterwarnings('ignore',category=UserWarning) #hacky to get tf quiet
#Tasks


#What would you do if you weren't afraid
#What would you do if money were no option
#What would you do if you knew you would succeed - start a company

#Environments, ntraj
#MountainCar,   [1,3,5,10,15] (120 steps/E_traj)
#Hopper         [
#Pong           [6,125,157,500] (1600 steps/E_traj)
#Enduro         [.1,1,20,80]               (10000 steps/E_traj)
#UpNDown        [
#DeHaan optimization: 10 epoch * 500 batch * 64 samp/epoch or = 320,000 steps
# GAIL Max performance:
# HalfCheetah       4850 #sac 10500, 5000000 steps
# Hopper            3615 #sac 3600 (70% good) 5000000 steps
# Walker            6965 #sac 5500, 5000000 steps, _2; HumanoidPyBullet settings 5000000 steps 5400
# Ant               4132 #sac 3900 (70% good) 5000000 steps
# Humanoid          10200
# Reacher           -5


env_id = ['CartPole-v1','Acrobot-v1','MountainCar-v0','PongNoFrameskip-v4','EnduroNoFrameskip-v4','UpNDownNoFrameskip-v4','Hopper??'][2]


# Train expert
if 0:
    for env_id in ['Humanoid-v2','Ant-v2','Reacher-v2','Hopper-v2','HalfCheetah-v2','Walker2d-v2']:
    #for env_id in ['CartPole-v1']:
        #ALICE.train_zoo_agent(env_id,'trpo','my_RL_experts',n_steps=1000000)
        #ALICE.train_zoo_agent(env_id,'dqn','my_RL_experts',n_steps=500000,env_kwargs={'max_episode_steps':2000})
        #ALICE.train_zoo_agent(env_id,'ppo2','my_RL_experts',n_steps=1000000)
        ALICE.train_zoo_agent(env_id,'trpo','my_RL_experts',n_steps=5000000)
        #ALICE.train_zoo_agent('Acrobot-v1','dqn','my_RL_experts',n_steps=1000000)
        #ALICE.train_zoo_agent('EnduroNoFrameskip-v4','dqn','logs',n_steps=15000000)
        #ALICE.train_zoo_agent('UpNDownNoFrameskip-v4','dqn','logs',n_steps=15000000)
        #subprocess.run(f'python {path}/train.py --algo {algo} --env {env_id} -n {n_RL}')
if 0:
     ALICE.train_zoo_agent('Humanoid-v2','sac','my_RL_experts',n_steps=15000000)
# Test expert
if 0:
    for env_id in ['Humanoid-v2']:
#        env_id = 'Walker2d-v2'
    #for env_id in ['HalfCheetah-v2','Hopper-v2','Ant-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']:
        model = ALICE.get_zoo_model(env_id,RL_algo='sac',exp_id=0,RL_expert_folder='my_RL_experts',load_best=True)
        pi_E = lambda obs :model.predict(obs)[0]
        traj_df = ALICE.get_trajectories(pi_E,env_id,N_traj=1,verbose=2,render=True,vec_env=True,T_max=1000)#,path='cached_experts/'+env_id+'-train.pkl.xz')
        print(len(traj_df),len(traj_df)/1000)

# Test one run
if 0:
    for env_id in ['CartPole-v1']:#,'CartPole-v1','MountainCar-v0',]:
    #for env_id in ['HalfCheetah-v2','Hopper-v2','Ant-v2','Humanoid-v2','Reacher-v2','Walker2d-v2'][:1]:
        res = ALICE.alg_runner('ALICE-FAIL',env_id,verbose=2,total_opt_steps=1000000,N_epoch=10,N_E_traj=10,add_history=False,N_agg_iter=20,N_FAIL_samps=3,#model_reg_coeff=1.0,#kill_feats=[4,5],#
                               random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,learning_rate_BC=0.1,learning_rate=1.0,density_ratio_feature_map='linear')
# Test one run
if 0:
    for env_id in ['HalfCheetah-v2']:#,'Hopper-v2','Ant-v2','Humanoid-v2','Reacher-v2','Walker2d-v2'][:1]:
        res = ALICE.alg_runner('BC',env_id,verbose=2,total_opt_steps=60000000,N_epoch=10,N_E_traj=25,add_history=False,N_agg_iter=5,
                            random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,learning_rate=0.00003,H_dims=[512],loss_function='mse',partial_obs=False)
# Run IL

'''#############################################################################
                                   Classic Control 
#############################################################################'''


# Run IL
if 0:
  env_id = 'Acrobot-v1'
  exp_name = 'exp1'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    for j in range(7):
      for i in [100,400]:
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts'))
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',run_seed=j))
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[4,5],results_path=results_path)
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    filters = {'final':True}
    lines = ['alg','density_ratio_feature_map']
    legend = ['Cov+H, linear','Cov+H, poly 2','Cov+H, poly 3','Cov, linear','Cov, poly 2','Cov, poly 3']
    ALICE.plot_results(results_df,'N_E_traj','reward',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward (Partial Obs)',incl_exp=True,incl_rand=True,save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss (Partial Obs)',save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','class_test',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss (Partial Obs)',save_dir='results',legend=legend)
    #plt.show()

if 0:
  env_id = 'CartPole-v1'
  exp_name = 'exp1'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    for j in range(7):
      for i in [100,400]:
        #ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',results_path=results_path)
        #ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',run_seed=j,results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.1,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.1,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',kill_feats=[3],results_path=results_path)
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    filters = {'final':True}
    lines = ['alg','density_ratio_feature_map']
    legend = ['Cov+H, linear','Cov+H, poly 2','Cov+H, poly 3','Cov, linear','Cov, poly 2','Cov, poly 3']
    ALICE.plot_results(results_df,'N_E_traj','reward',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward (Partial Obs)',incl_exp=True,incl_rand=True,save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss (Partial Obs)',save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','class_test',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss (Partial Obs)',save_dir='results',legend=legend)

if 1:
  env_id = 'CartPole-v1'
  exp_name = 'exp01'
  results_path = 'results/results-'+env_id+'--exp1.csv'
  if 1:    
    results_df = pd.concat([ALICE.load_agg_save('results/results-'+env_id+'--exp0.csv',[]),ALICE.load_agg_save('results/results-'+env_id+'--exp1.csv',[])],ignore_index=True)
    filters = {'final':True,'density_ratio_feature_map':[None,'linear'],'kill_feats':[(),(3)],}
    lines = ['alg']
    legend = None#['Cov+H, linear','Cov+H, poly 2','Cov+H, poly 3','Cov, linear','Cov, poly 2','Cov, poly 3']
    ALICE.plot_results(results_df,'N_E_traj','reward',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward (Partial Obs)',incl_exp=True,incl_rand=True,save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss (Partial Obs)',save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','class_test',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss (Partial Obs)',save_dir='results',legend=legend)
    #plt.show()

if 0:
  env_id = 'MountainCar-v0'
  exp_name = 'exp1'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    for j in range(7):
      for i in [100,400]:
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts'))
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',run_seed=j))
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.03,random_demos=False,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.03,random_demos=False,density_ratio_feature_map='linear',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.03,random_demos=False,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.03,random_demos=False,density_ratio_feature_map='poly 2',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.03,random_demos=False,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
        ALICE.alg_runner('ALICE-Cov',env_id,verbose=0,total_opt_steps=4000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.03,random_demos=False,density_ratio_feature_map='poly 3',
                         N_agg_iter=20,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,results_path=results_path,kill_feats=[0])
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    filters = {'final':True}
    lines = ['alg','density_ratio_feature_map']
    legend = ['Cov+H, linear','Cov+H, poly 2','Cov+H, poly 3','Cov, linear','Cov, poly 2','Cov, poly 3']
    ALICE.plot_results(results_df,'N_E_traj','reward',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward (Partial Obs)',incl_exp=True,incl_rand=True,save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss (Partial Obs)',save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','class_test',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss (Partial Obs)',save_dir='results',legend=legend)
    #plt.show()


'''#############################################################################
                                   Mujoco 
#############################################################################'''

if 0:
  env_id = 'Hopper-v2'
  exp_name = 'exp2'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    for j in range(1):
      for i in [1,5,20,100,200]:
        
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=20000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.003,random_demos=False,results_path=results_path,
                         N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,H_dims=[512,256])
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=20000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.003,random_demos=False,results_path=results_path,
                         N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,H_dims=[512,256])
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts'))
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',run_seed=j))
  if 0:
    results_df = ALICE.load_agg_save(results_path,[])
    ALICE.plot_results(results_df,'N_E_traj','reward',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward',incl_exp=True,incl_rand=True,save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss',save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','class_test',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss',save_dir='results')
    #plt.show()

if 0:
  env_id = 'Reacher-v2'
  exp_name = 'exp0'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    for j in range(1,7):
      for i in [1,5,20,100]:
        
        ALICE.load_agg_save(results_path,ALICE.alg_runner('BC',env_id,verbose=2,total_opt_steps=2000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.0003,random_demos=False,
                                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,H_dims=[512,128,32]))
        ALICE.load_agg_save(results_path,ALICE.alg_runner('BC',env_id,verbose=2,total_opt_steps=2000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.0003,random_demos=False,
                                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False,H_dims=[512,128,32]))
        #ALICE.load_agg_save(results_path,ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts'))
        ALICE.load_agg_save(results_path,ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,add_history=False,RL_expert_folder='my_RL_experts',run_seed=j))
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    ALICE.plot_results(results_df,'N_E_traj','reward',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward',incl_exp=True,incl_rand=True,save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss',save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','class_test',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss',save_dir='results')
    #plt.show()

if 0:
  #exp0 try different loss funcs and earning rates
  env_id = 'HalfCheetah-v2'
  exp_name = 'exp1'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    for n in [4,11,18,25]:
    #ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=60000000,N_epoch=10,N_E_traj=25,add_history=False,N_agg_iter=10,random_demos=False,RL_expert_folder='my_RL_experts',
    #                 RL_expert_load_best=True,learning_rate=0.0001,H_dims=[512],loss_function='mse',partial_obs=False,results_path=results_path)
    #ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=60000000,N_epoch=10,N_E_traj=25,add_history=False,N_agg_iter=10,random_demos=False,RL_expert_folder='my_RL_experts',
    #                 RL_expert_load_best=True,learning_rate=0.0003,H_dims=[512],loss_function='mse',partial_obs=False,results_path=results_path)
      ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=True,N_agg_iter=1,random_demos=False,RL_expert_folder='my_RL_experts',
                     RL_expert_load_best=True,learning_rate=0.003,H_dims=[512],loss_function='L2',partial_obs=False,results_path=results_path,render_final=False)

if 0:
  #exp0 try different loss funcs and earning rates
  env_id = 'HalfCheetah-v2'
  exp_name = 'exp1'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    for n in [11,18,25,50,100,200]:
      ALICE.alg_runner('Expert',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=False,N_agg_iter=10,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,
                     RL_expert_load_best=True,partial_obs=False,results_path=results_path)
      ALICE.alg_runner('Expert',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=False,N_agg_iter=10,RL_expert_folder='my_RL_experts',RL_expert_exp_id=2,
                     RL_expert_load_best=True,partial_obs=False,results_path=results_path)
      #ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=False,N_agg_iter=1,random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,
      #               RL_expert_load_best=True,learning_rate=0.003,H_dims=[512],loss_function='L2',partial_obs=False,results_path=results_path,render_final=False)
      #ALICE.alg_runner('ALICE-Cov',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=False,N_agg_iter=10,random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,
      #               RL_expert_load_best=True,learning_rate=0.003,H_dims=[512],loss_function='L2',partial_obs=False,results_path=results_path,render_final=False)
      #ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=True,N_agg_iter=1,random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,
      #               RL_expert_load_best=True,learning_rate=0.003,H_dims=[512],loss_function='L2',partial_obs=False,results_path=results_path,render_final=False)
      #ALICE.alg_runner('ALICE-Cov',env_id,verbose=4,total_opt_steps=10000000,N_epoch=10,N_E_traj=n,add_history=True,N_agg_iter=10,random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,
      #               RL_expert_load_best=True,learning_rate=0.003,H_dims=[512],loss_function='L2',partial_obs=False,results_path=results_path,render_final=False)
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    filters = {'final':True}
    lines = ['alg','RL_expert_exp_id']
    legend = None#['Cov Fast','Cov Slow','Cov+H Fast','Cov+H Slow','BC Fast','BC Slow','BC+H Fast','BC+H Slow']
    ALICE.plot_results(results_df,'N_E_traj','reward',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward',incl_exp=True,incl_rand=True,save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss',save_dir='results',legend=legend)
    ALICE.plot_results(results_df,'N_E_traj','class_test',lines,filters,env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss',save_dir='results',legend=legend)

'''#############################################################################
                                   Atari 
#############################################################################'''

if 0:
    env_id = 'EnduroNoFrameskip-v4'
    exp_name = 'great_same_expert'
    results_path = 'results-'+env_id+'--'+exp_name+'.csv'
    results_list = []
    for j in range(1):
      for i in [1,10,200,800]:
        ALICE.load_agg_save(results_path,ALICE.alg_runner('Expert',env_id,verbose=0,N_E_traj=i,RL_expert_folder=None,T_max=1000))
        ALICE.load_agg_save(results_path,ALICE.alg_runner('Random',env_id,verbose=0,N_E_traj=i,RL_expert_folder=None,T_max=1000,run_seed=j))
        ALICE.load_agg_save(results_path,ALICE.alg_runner('BC',env_id,verbose=0,N_E_traj=i,RL_expert_folder=None,T_max=1000,
                                                          total_opt_steps=2400000,learning_rate=0.0003,N_agg_iter=4,opt_seed=j,run_seed=j,add_history=False))
if 0:
  env_id = 'PongNoFrameskip-v4'
  exp_name = 'exp0'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    for j in range(1):
      for i in [6,10,125]:
        ALICE.load_agg_save(results_path,ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=3000000,N_epoch=10,N_E_traj=i,add_history=True,learning_rate=0.0003,random_demos=False,
                                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False))
        ALICE.load_agg_save(results_path,ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=3000000,N_epoch=10,N_E_traj=i,add_history=False,learning_rate=0.0003,random_demos=False,
                                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',partial_obs=False))
        
        
  if 1:
    results_df = ALICE.load_agg_save(results_path,[])
    ALICE.plot_results(results_df,'N_E_traj','reward',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Reward',incl_exp=True,incl_rand=True,save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Train Loss',save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','class_test',['alg'],filters={'final':True},env_id=env_id,exp_name=exp_name,xscale='log',title=env_id+' Test Classification Loss',save_dir='results')
    #plt.show()

if 0:
    env_id = 'PongNoFrameskip-v4'
    exp_name = 'great_same_expert'
    results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
    results_df = ALICE.load_agg_save(results_path,[])
    #results_df = pd.concat(results_list,ignore_index=True)
    ALICE.plot_results(results_df,'N_E_traj','reward',['alg','total_opt_steps'],filters={'total_opt_steps':[600000,1200000,1800000,2400000]},
                       env_id =env_id,exp_name=exp_name,xscale='log',title='Atari Pong Reward - Intermediate Results',incl_exp=True,incl_rand=True,save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg','total_opt_steps'],filters={'total_opt_steps':[600000,1200000,1800000,2400000]},
                       env_id =env_id,exp_name=exp_name,xscale='log',title='Atari Pong Train Loss - Intermediate Results',save_dir='results')
    ALICE.plot_results(results_df,'N_E_traj','class_test',['alg','total_opt_steps'],filters={'total_opt_steps':[600000,1200000,1800000,2400000]},
                       env_id =env_id,exp_name=exp_name,xscale='log',title='Atari Pong Test Classification Loss - Intermediate Results',save_dir='results')
    #ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg'],filters=None,env_id=env_id,exp_name=exp_name)
    #ALICE.plot_results(results_df,'N_E_traj','loss_test',['alg'],filters=None,env_id=env_id,exp_name=exp_name)
    #ALICE.plot_results(results_df,'N_E_traj','class_test',['alg'],filters=None,env_id=env_id,exp_name=exp_name)
    #plt.show()
    

#Jonathan own the project (high level direction) 
# Treat this as your baby. 
# Pair code (accountability session)
# We should expect either better or equal performance by adding features


import ALICE, os, subprocess
import matplotlib.pyplot as plt
import pandas as pd

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import warnings
#warnings.filterwarnings('ignore',category=FutureWarning) #hacky to get tf quiet
#warnings.filterwarnings('ignore',category=UserWarning) #hacky to get tf quiet

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
    for env_id in ['CartPole-v1']:
#        env_id = 'Walker2d-v2'
    #for env_id in ['HalfCheetah-v2','Hopper-v2','Ant-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']:
        model = ALICE.get_zoo_model(env_id,RL_algo='dqn',exp_id=0,RL_expert_folder='my_RL_experts',load_best=True)
        pi_E = lambda obs :model.predict(obs)[0]
        traj_df = ALICE.get_trajectories(pi_E,env_id,N_traj=100,verbose=2,render=False,vec_env=True,T_max=1000,path='cached_experts/'+env_id+'-train.pkl.xz')
        print(len(traj_df),len(traj_df)/1000)

# Test one run
if 1:
    for env_id in ['CartPole-v1']:#,'CartPole-v1','MountainCar-v0',]:
    #for env_id in ['HalfCheetah-v2','Hopper-v2','Ant-v2','Humanoid-v2','Reacher-v2','Walker2d-v2'][:1]:
        res = ALICE.alg_runner('ALICE-FAIL',env_id,verbose=2,total_opt_steps=500000,N_epoch=10,kill_feats=[3],entropy_coeff = 0,pair_with_E=False,
                               N_E_traj=30,N_ALICE_traj=1,add_history=False,N_agg_iter=5,N_FAIL_samps=3,results_path='test.csv',#model_reg_coeff=1.0,
                               random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,opt_seed=0,run_seed=0,recent_samp_priority_exp=1,
                               learning_rate_BC=0.1,learning_rate=.003,density_ratio_feature_map='linear',adversary_feature_map='standardscaler rff-256')#,horizon_weight_offset_exp = (0,500,0))#(5,5,2))
    #res = pd.concat([res1,res2],ignore_index=True)
    lines = ['alg']
    ALICE.plot_results(res,'iteration_num','JS_div',lines)
    #ALICE.plot_results(res,'iteration_num','entropy',lines)
    ALICE.plot_results(res,'iteration_num','reward',lines)
    ALICE.plot_results(res,'iteration_num','loss_train',lines)
    #ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines)
    #ALICE.plot_results(res,'iteration_num','loss_test',lines)
    plt.show()

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

if 0:
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
import multiprocessing as mp
####### Entropy tests
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'priority10'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    with mp.Pool(12) as p:
      for ec in [0,.5]:
        for pe in [1,.9,.8,.7]:
          #ALICE.alg_runner('ALICE-FAIL',env_id,verbose=2,total_opt_steps=2000000,N_epoch=10,kill_feats=[3],entropy_coeff = ec,
          #                     N_E_traj=20,add_history=False,N_agg_iter=20,N_FAIL_samps=3,#model_reg_coeff=mr,
          #                     random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,opt_seed=0,run_seed=0,
          #                     learning_rate_BC=0.1,learning_rate=lr,density_ratio_feature_map='linear',adversary_feature_map='linear',results_path=results_path)
          p.apply_async(ALICE.alg_runner,('ALICE-FAIL',env_id),{'verbose':0,'total_opt_steps':2000000,'N_agg_iter':20,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':ec,
                                                                'N_E_traj':10,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
                                                                'learning_rate_BC':0.1,'learning_rate':0.001,'density_ratio_feature_map':'linear','adversary_feature_map':'linear',
                                                                'results_path':results_path,'recent_samp_priority_exp':pe})
      p.close(); p.join()
    
  if 1:
    res = ALICE.load_agg_save_safe(results_path,[])
    lines = ['alg','entropy_coeff','recent_samp_priority_exp']
    filters = None
    save_dir = None #'results'
    cmap = 'tab20'
    ALICE.plot_results(res,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    plt.show()
    
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'priority10_2'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    with mp.Pool(12) as p:
      for lr in [.001,.0001]:
        for pe in [1,.6]:
          for ec in [0,.5]:
          #ALICE.alg_runner('ALICE-FAIL',env_id,verbose=2,total_opt_steps=2000000,N_epoch=10,kill_feats=[3],entropy_coeff = ec,
          #                     N_E_traj=20,add_history=False,N_agg_iter=20,N_FAIL_samps=3,#model_reg_coeff=mr,
          #                     random_demos=False,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,opt_seed=0,run_seed=0,
          #                     learning_rate_BC=0.1,learning_rate=lr,density_ratio_feature_map='linear',adversary_feature_map='linear',results_path=results_path)
              p.apply_async(ALICE.alg_runner,('ALICE-FAIL',env_id),{'verbose':0,'total_opt_steps':4000000,'N_agg_iter':40,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':ec,
                                                                    'N_E_traj':10,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
                                                                    'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear','adversary_feature_map':'linear',
                                                                    'results_path':results_path,'recent_samp_priority_exp':pe})
      p.close(); p.join()
    
  if 1:
    res = ALICE.load_agg_save_safe(results_path,[])
    lines = ['alg','learning_rate','recent_samp_priority_exp','entropy_coeff']
    filters = None
    save_dir = None #'results'
    cmap = 'tab20'
    ALICE.plot_results(res,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    plt.show()

if 0:
  env_id = 'CartPole-v1'
  exp_name = 'priority10_3-different-demos'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    with mp.Pool(12) as p:
      for lr in [.001]:
        for pe in [1,.9]:
          for ec in [0,.01]:
            p.apply_async(ALICE.alg_runner,('ALICE-FAIL',env_id),{'verbose':0,'total_opt_steps':3000000,'N_agg_iter':30,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':ec,
                                                                    'N_E_traj':30,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
                                                                    'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear','adversary_feature_map':'linear',
                                                                    'results_path':results_path,'recent_samp_priority_exp':pe})
      p.close(); p.join()
    
  if 1:
    res = ALICE.load_agg_save_safe(results_path,[])
    lines = ['alg','recent_samp_priority_exp','entropy_coeff']
    filters = None
    save_dir = 'results'
    cmap = 'tab20'
    ALICE.plot_results(res,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'priority10_3-different-demos-meanstdpoly2'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    with mp.Pool(12) as p:
      for lr in [.001]:
        for pe in [1,.9]:
          for ec in [0,.01]:
            p.apply_async(ALICE.alg_runner,('ALICE-FAIL',env_id),{'verbose':0,'total_opt_steps':2000000,'N_agg_iter':20,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':ec,
                                                                    'N_E_traj':30,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
                                                                    'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear','adversary_feature_map':'standardscaler poly-2',
                                                                    'results_path':results_path,'recent_samp_priority_exp':pe})
      p.close(); p.join()
    
  if 1:
    res = ALICE.load_agg_save_safe(results_path,[])
    lines = ['alg','recent_samp_priority_exp','entropy_coeff']
    filters = None
    save_dir = 'results'
    cmap = 'tab20'
    ALICE.plot_results(res,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
 
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'horizon10'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    with mp.Pool(3) as p:
      for lr in [.001]:
        for pe in [1]:
          for hwoe in [(0,500,0),(10,10,1),(5,5,1.5),(5,5,2)]:
            #p.apply_async(ALICE.alg_runner,('ALICE-FAIL',env_id),{'verbose':0,'total_opt_steps':3000000,'N_agg_iter':30,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':0,
            #                                                        'N_E_traj':10,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
            #                                                        'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear','adversary_feature_map':'linear',
            #                                                        'results_path':results_path,'recent_samp_priority_exp':pe,'horizon_weight_offset_exp':hwoe,'pair_with_E':True})
            ALICE.alg_runner('ALICE-FAIL',env_id,**{'verbose':0,'total_opt_steps':4000000,'N_agg_iter':40,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':0,
                                                                    'N_E_traj':10,'add_history':False,'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,
                                                                    'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear','adversary_feature_map':'linear',
                                                                    'results_path':results_path,'recent_samp_priority_exp':pe,'horizon_weight_offset_exp':hwoe,'pair_with_E':True})
      p.close(); p.join()
    
  if 1:
    res = ALICE.load_agg_save_safe(results_path,[])
    lines = ['alg','horizon_weight_offset_exp']
    filters = None
    save_dir = 'results'
    cmap = 'tab10'
    ALICE.plot_results(res,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(res,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap) 
    
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'dropfirst-10'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 0:
    with mp.Pool(3) as p:
      for lr in [.003]:
        for pe in [1]:
          for df in [0,2,4,8,16]:
            args = ('ALICE-FAIL',env_id)
            kwargs = {'verbose':2,'total_opt_steps':1000000,'N_agg_iter':10,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':0,'N_E_traj':10,'add_history':False,'drop_first':df,
                      'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear',
                      'adversary_feature_map':'linear','results_path':results_path,'recent_samp_priority_exp':pe,'horizon_weight_offset_exp':None,'pair_with_E':True}
            ALICE.alg_runner(*args,**kwargs)
            #p.apply_async(ALICE.alg_runner,args,kwargs)
      p.close(); p.join()
    
  if 1:
    lines = ['alg','drop_first']
    filters = None
    save_dir = 'results'
    cmap = 'tab10'
    ALICE.plot_results(results_path,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap) 
    #plt.show()
if 0:
  env_id = 'CartPole-v1'
  exp_name = 'n_ALICE-30'
  results_path = 'results/results-'+env_id+'--'+exp_name+'.csv'
  if 1:
    with mp.Pool(4) as p:
      for lr in [.003]:
        for pe in [1]:
          for NA in [1,5,10,30]:
            args = ('ALICE-FAIL',env_id)
            kwargs = {'verbose':2,'total_opt_steps':3000000,'N_agg_iter':30,'N_epoch':10,'kill_feats':(3,),'entropy_coeff':0,'N_E_traj':30,'add_history':False,'N_ALICE_traj':NA,
                      'N_FAIL_samps':2,'RL_expert_folder':'my_RL_experts','opt_seed':0,'run_seed':0,'learning_rate_BC':0.1,'learning_rate':lr,'density_ratio_feature_map':'linear',
                      'adversary_feature_map':'linear','results_path':results_path,'recent_samp_priority_exp':pe,'horizon_weight_offset_exp':None,'pair_with_E':True}
            #ALICE.alg_runner(*args,**kwargs)
            p.apply_async(ALICE.alg_runner,args,kwargs)
      p.close(); p.join()
    
  if 1:
    lines = ['alg','N_ALICE_traj']
    filters = None
    save_dir = 'results'
    cmap = 'tab20c'
    ALICE.plot_results(results_path,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap) 

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

if 0:
    ALICE.verify_adversary(d=2,feature_map_pipeline='standardscaler poly-2')

if 0: #Old ALICE bottom text
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
        #alg_runner('DaD',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_agg_iter=10,N_E_traj=10)
        #alg_runner('BC',env_id,verbose=3,total_opt_steps=1000000,N_epoch=20,resume_BC=False,N_E_traj=1)
        df = alg_runner('Expert-FAIL',env_id,verbose=5,total_opt_steps=10000000,N_epoch=4,resume_BC=False,N_E_traj=2,density_ratio_feature_map='poly-1', adversary_feature_map='rff-2048 rff-512',add_history=False,pi0_BC_init=False,N_agg_iter=10,opt_seed=0,run_seed=0,linear=True,kill_feats=None,learning_rate=0.0001,NN_mid_as_feats=True,N_FAIL_samps=3)
        print(df)
    if 0:
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
    if 1:
        X = tf.cast(np.random.rand(5,2),tf.float32)
        model = lambda x : X
        model_prob_a = lambda model_out,a: tf.gather_nd(tf.nn.softmax(model_out,axis=-1),a,batch_dims=1)
        a = [[0],[1],[1],[0],[1]]
        print(X,a,tf.nn.softmax(X,axis=-1))
        print(batch_model_prob_a(model_prob_a,model,X,a,batch_size=2))

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
    
    # Sanjiban 
    # 1) Track the number of misclassified samples - track on-policy validation accuracy
    # Why does the reward look so good 
    # Adding history is not bad. It a
    # How to test it's overfitting to the exceptions vs generalizing to the exceptions:
    # For all features you map back to the same action.
    # Adding more features should help me do better. Prove me wrong.


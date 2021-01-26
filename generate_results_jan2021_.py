import ALICE
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
from itertools import product
exp_name = 'BC'
results_dir = 'results_jan2021/'
results_path = lambda env_id : 'results_jan2021/results-'+env_id+'--'+exp_name+'.csv'

def print_results(env_ids,results_dir,exp_name,algs=['BC','BC+H'],params = None, filters = None,final=True,latex_table=False,data='reward'):
    # Format filters so all values are lists
    filters = {k:([filters[k]] if type(filters[k]) is not list else filters[k]) for k in filters} if filters is not None else dict()
    params = [params] if type(params) is str else params if params is not None else []
    cols = list(set(params) | set(filters.keys()))
    
    # Build header string and calculate width
    N_cols = len(cols)
    N_alg = len(algs)
    alg_hdrs = [alg + ' (# avg)' for alg in algs] 
    if latex_table:
        hdr_str = (' {:<20} &' + ' {:^15} &'*N_cols +' {:^14} &'+' {:^18} &'*N_alg).format('Environment',*[k[:15] for k in cols],'Expert (rew)',*algs_hdrs)
    else:
        hdr_str = ('| {:<20} |' + ' {:^15} |'*N_cols +' {:^14} |'+' {:^18} |'*N_alg).format('Environment',*[k[:15] for k in cols],'Expert (rew)',*alg_hdrs)
    w = len(hdr_str)
    
    print('-'*w+'\n|{0:^{1}}|\n'.format(data,w-2)+'-'*w)
    print(hdr_str)
    print('-'*w)
    for env_id in env_ids:
        if env_id[-2]=='_':
            df = ALICE.load_agg_save(results_dir+'results-'+env_id[:-2]+'--'+exp_name+'.csv')
        else:
            df = ALICE.load_agg_save(results_dir+'results-'+env_id+'--'+exp_name+'.csv')
        if final:
            df = df[df['final']==True]
        train_r_mean,train_r_std = df.loc[df.index[0],['train_r_mean','train_r_std']]
        
        lines_df = df[cols].drop_duplicates().dropna()
        if len(filters)>0:
            print('Filtered')
            df = df[(df[filters.keys()].isin(filters)).all(axis=1)]
        for i,line in lines_df.iterrows():
            line_str = ['| {:<20} |'+' {:^15} |'*N_cols + '{:>8.0f}+-{:<5.0f} |',
                        ' {:<20} '+'& {:^15} '*N_cols + '& ${:.1f}\pm {:.1f}$ '][latex_table].format(
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
                    data_str = ['{:>8.0f}+-{:<5.0f} ({}) |','& ${:.1f}\pm {:.1f} ({})$ '][latex_table].format(data_mean,pooled_std,len(data_vec))
                else:
                    data_str = ['{:^15.5g} ({}) |','& ${:.5f} ({})$ '][latex_table].format(data_mean,len(data_vec))
                line_str += data_str
            print(line_str)
    print('-'*w)

# 
if 0:
    for env_id in ['CartPole-v1','Acrobot-v1','MountainCar-v0']:
        for j in range(7):
            ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=500000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.01,H_dims=(64,),
                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',results_path=results_path(env_id))
            ALICE.alg_runner('BC',env_id,verbose=4,total_opt_steps=500000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.01,H_dims=(64,),
                             N_agg_iter=5,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',results_path=results_path(env_id))
if 0:
    env_id = 'HalfCheetah-v2'
    for j in range(7):
        #exp_id = 2 is Slow Cheetah
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=10000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=2,H_dims=[512,128,32])
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=10000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=2,H_dims=[512,128,32])
if 0:
    env_id = 'Walker2d-v2' #BBC2
    for j in range(7):
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=30000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.0003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,128,32])
        ALICE.alg_runner('BC',env_id,verbose=3,total_opt_steps=30000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.0003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,128,32])

if 1:
    env_id = 'Ant-v2'
    for j in range(1):
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,512])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,512])

#512,512,001,20M - 2780, 3163 (.0204/.1803)
#256,256,001,20M - 2590, 2440 (.0410/.1941)
#128,128,001,20M - 3454, 2981 (.0903/.1981)
#100,100,001,40M - 3617, 2767 (.1056/.1989)
#100,100,001,20M - 3712, 3000 (.1147/.2013)*
#100,100,003,20M - 2555, 2774 (.1303/.2271)
#100,100,0002,20M - 2611, 3095 (.1745/.2308)
#64,64,001,20M - 3062, 3202 (.1685/.2298)
#64,64,0002,20M - 2811, 2634 (.1876/.2385)
#64,64,001,40M - 3318, 3256 (.1608/.2260)

if 1:
    env_id = 'Hopper-v2'
    for j in range(1):
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=2,opt_seed=j,run_seed=j,RL_expert_folder='my_RL_experts',H_dims=[512,16,512])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=2,opt_seed=j,run_seed=j,RL_expert_folder='my_RL_experts',H_dims=[512,16,512])
          
print_results(env_ids=['Hopper-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BC',data='loss_train')
print_results(env_ids=['Hopper-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BC',data='reward')
#print_results(env_ids=['Hopper-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BBC',data='reward',final=False)
#print_results(env_ids=['Hopper-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BBC',data='loss_train',final=False)
#1024,1024,001,20M-1734,1981(.0064,.0264)
#750,750,001,20M - 2897,1624(.0066,.0266)
#512,512,001,20M - 3258,2057(.0075,.0276)* 2622+-759,1519+-673
#256,256,001,20M - 1441,985 (.0135,.0335)
#128,128,001,20M - 2118,54  (.0258,.0453)
#100,100,001,20M - 542,991  (.0335,.0512)
#64,64,001,20M - 218,236    (.0584,.0739)

if 0:
    env_id = 'Walker2d-v2'
    HD = 512
    for j in range(7):
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=30000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=3,opt_seed=j,run_seed=j,RL_expert_folder='my_RL_experts',H_dims=[HD,HD])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=30000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.001,results_path=results_path(env_id),
                         N_agg_iter=3,opt_seed=j,run_seed=j,RL_expert_folder='my_RL_experts',H_dims=[HD,HD])
#print_results(env_ids=['Walker2d-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BC',data='reward',final=True)
#print_results(env_ids=['Walker2d-v2'],results_dir=results_dir,params=['total_opt_steps','H_dims','learning_rate'],exp_name='BBC',data='reward',final=False)
df = ALICE.load_agg_save(results_path('Hopper-v2'))
plt.scatter(df['reward'].to_numpy(),df['loss_train'].to_numpy())
plt.show()

# Walker
#512,512,001,30M - 4209,4451 (.0056,.0281)* 4445+-2054, 3843+-2104
#256,256,001,30M - 2297,5350 (.0098,.0351)
#128,128,001,30M - 5514,1212 (.0224,.0474)
#100,100,001,30M - 0,2722.5 (.0332,.0548)
#64,64,001,30M - 1897,1443 (.0502,.0713)


if 0:
    for env_id in ['Hopper-v2','Walker2d-v2','Ant-v2']:
        for j in range(1):
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.0003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,128,32])
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.0003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[512,128,32])
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[256,64])
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[256,64])
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.0003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[256,64])
            ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.0003,results_path=results_path(env_id),
                             N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',H_dims=[256,64])

if 0:
    env_id = 'HalfCheetah-v2'
    for j in range(1):
        # Fast Cheetah
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,H_dims=[512,128,32])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=1,H_dims=[512,128,32])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=False,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=2,H_dims=[512,128,32])
        ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=20000000,N_epoch=10,N_E_traj=25,add_history=True,learning_rate=0.003,results_path=results_path(env_id),
                         N_agg_iter=10,opt_seed=j,run_seed=j,linear=False,RL_expert_folder='my_RL_experts',RL_expert_exp_id=2,H_dims=[512,128,32])

    
# Things noticed: Ant, Hopper, Walker flatline at highish variance pretty quickly Walker probably needs 20M steps for +H
#   Will try slower learning rate (initially tried .003)
#   Will try simpler model (256,64) rather than 
    

# do 50% BC
# Exploit structure with yuge data

#print_results(env_ids=['CartPole-v1','Acrobot-v1','MountainCar-v0','Ant-v2','HalfCheetah-v2_2','Walker2d-v2'],results_dir=results_dir,exp_name=exp_name,latex_table=True)
#print_results(env_ids=['CartPole-v1','Acrobot-v1','MountainCar-v0'],results_dir=results_dir,exp_name=exp_name,filters={'H_dims':['(512,)','(64,)','(32,)','(16,)']})
#print_results(env_ids=['Hopper-v2','Reacher-v2','Walker2d-v2','Ant-v2'],results_dir=results_dir,exp_name=exp_name,final=True,filters={'H_dims':['(512, 128, 32)','(256, 64)'],'learning_rate':[0.003,0.0003]},latex_table=False)
#print_results(env_ids=['Ant-v2'],results_dir=results_dir,exp_name=exp_name,final=False,latex_table=False,filters={'H_dims':['(512, 128, 32)','(256, 64)','(128, 32)','(64, 64, 64, 64)','(64, 8, 64)'],'learning_rate':[0.003,0.0003],'iteration_num':[*range(10)]})
#ALICE.plot_results(results_path('Ant-v2'),'iteration_num','reward',['H_dims','learning_rate','alg'],None,exp_name=exp_name)
#plt.show()
#print_results(env_ids=['Hopper-v2','Reacher-v2','Walker2d-v2','Ant-v2'],results_dir=results_dir,exp_name=exp_name,final=True,filters={'iteration_num':[*range(10)]},latex_table=False)
#print_results(env_ids=['Hopper-v2','Reacher-v2','Walker2d-v2','Ant-v2','HalfCheetah-v2_1','HalfCheetah-v2_2'],results_dir=results_dir,exp_name=exp_name,final=False,filters=)
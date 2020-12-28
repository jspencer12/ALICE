import ALICE
import matplotlib.pyplot as plt
import multiprocessing as mp

if 0:
  env_id = 'CartPole-v1'
  exp_name = 'initial_param_exploration'
  results_path = 'results_dec2020/results-'+env_id+'--'+exp_name+'.csv'
  
  if 0:
    with mp.Pool(4) as p:
      for drfm in ['RuLSIF','standardscaler poly-2','standardscaler poly-2 RuLSIF','standardscaler RuLSIF']:
        for tos in [50000,100000]:
          for dra in [.5,1]:
            args = ('ALICE-Cov',env_id)
            kwargs = {'N_E_traj':100,'verbose':2,'total_opt_steps':tos,'kill_feats':(3,),'add_history':False,'RL_expert_folder':'my_RL_experts','FORWARD':True,
                      'opt_seed':0,'run_seed':0,'learning_rate':0.01,'density_ratio_feature_map':drfm,'density_ratio_alpha':dra,'results_path':results_path}
            #ALICE.alg_runner(*args,**kwargs)
            p.apply_async(ALICE.alg_runner,args,kwargs)
      p.close(); p.join()
    
  if 1:
    lines = ['density_ratio_feature_map','total_opt_steps','density_ratio_alpha']
    filters = None
    save_dir = 'results'
    cmap = 'tab20c'
    #ALICE.plot_results(results_path,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'iteration_num','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    plt.show()

# Good parameters: 'density_ratio_feature_map':'standardscaler poly-2', 'density_ratio_alpha':1, 'total_opt_steps':100000

if 1:
  env_id = 'CartPole-v1'
  exp_name = 'benchmark'
  results_path = 'results_dec2020/results-'+env_id+'--'+exp_name+'.csv'
  
  if 0:
    with mp.Pool(4) as p:
      for hist in [False,True]:
        for seed in range(5):
          for alg in ['BC','ALICE-Cov']:
            for N in [10,50,100,500]:
                args = (alg,env_id)
                kwargs = {'N_E_traj':N,'verbose':0,'total_opt_steps':200000,'kill_feats':(3,),'add_history':hist,'RL_expert_folder':'my_RL_experts','FORWARD':True,
                          'opt_seed':seed,'run_seed':seed,'learning_rate':0.01,'density_ratio_feature_map':'standardscaler poly-2','density_ratio_alpha':1,'results_path':results_path}
                #ALICE.alg_runner(*args,**kwargs)
                p.apply_async(ALICE.alg_runner,args,kwargs)
      p.close(); p.join()
    
  if 1:
    lines = ['alg']
    filters = None
    save_dir = 'results'
    cmap = 'tab10'
    #ALICE.plot_results(results_path,'iteration_num','JS_div',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','entropy',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    ALICE.plot_results(results_path,'N_E_traj','reward',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','hindsight_loss_train',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    #ALICE.plot_results(results_path,'iteration_num','loss_test',lines,filters,save_dir=save_dir,exp_name=exp_name,colormap=cmap)
    plt.show()

import ALICE
import matplotlib.pyplot as plt

#Tasks
# 1) Remove final state
# 2) Crush DeHaan w/ BC
# 3) Fix FAIL??? How2?


#Environments, ntraj
#MountainCar,   [1,3,5,10,15] (120 steps/E_traj)
#Hopper         [
#Pong           [6,125,157,500] (1600 steps/E_traj)
#Enduro         [.1,1,20,80]               (10000 steps/E_traj)
#UpNDown        [


env_id = 'MountainCar-v0'
experiment_name = 'break_BC'
results_path = 'results-'+env_id+'--'+experiment_name+'.csv'
results_list = []

#Train expert


#Test expert

model_E = ALICE.get_expert_model(env_id)
pi_E = lambda obs: model_E.predict(obs)[0] #model_E.predict returns action,state for recurrent policies
df_E = ALICE.get_trajectories(pi_E,env_id,verbose=2,render=True,framestack_env=True)


if 0:
    for i in [1,3,5,10,15]:
      for j in range(1):
        results_list.append(ALICE.alg_runner('BC',env_id,verbose=0,total_opt_steps=1000000,N_epoch=3,N_E_traj=i,add_history=False,N_agg_iter=5,opt_seed=j,run_seed=j,linear=True))
    results_df = ALICE.load_agg_save(results_path,results_list)
    #results_df = pd.concat(results_list,ignore_index=True)
    ALICE.plot_results(results_df,'N_E_traj','reward','alg',filters=None,env_id =env_id,experiment_name=experiment_name)
    ALICE.plot_results(results_df,'N_E_traj','loss_train',['alg'],filters=None,env_id=env_id,experiment_name=experiment_name)
    ALICE.plot_results(results_df,'N_E_traj','loss_test',['alg'],filters=None,env_id=env_id,experiment_name=experiment_name)
    plt.show()

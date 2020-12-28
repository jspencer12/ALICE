import ALICE
import importlib
importlib.reload(ALICE)
import matplotlib.pyplot as plt
seed = 314
# RuLSIF
save_path = 'simple_results.csv'
res = ALICE.alg_runner('ALICE-Cov','CartPole-v1', verbose=2, total_opt_steps=50000, N_epoch=10, kill_feats=[3],
                       N_E_traj=100, add_history=False, N_FAIL_samps=2, entropy_coeff = 0,
                       results_path=save_path,RL_expert_folder='my_RL_experts',RL_expert_load_best=True,
                       opt_seed=seed, run_seed=seed, learning_rate=.01,
                       density_ratio_feature_map='standardscaler poly-2',FORWARD=True)

lines = ['alg']
#ALICE.plot_results(res,'iteration_num','entropy',lines)
ALICE.plot_results(save_path,'iteration_num','reward',lines)
ALICE.plot_results(save_path,'iteration_num','loss_train',lines)
#ALICE.plot_results(res,'iteration_num','hindsight_loss_train',lines)
#ALICE.plot_results(res,'iteration_num','loss_test',lines)
ALICE.plot_results(save_path,'iteration_num','loss_test',lines)
plt.show()
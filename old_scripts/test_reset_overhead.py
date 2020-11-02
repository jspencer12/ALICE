import gym

def reset_env(env,state=None,state_prev=None,action_prev=None):
    env_id = env.envs[0].unwrapped.spec.id if hasattr(env,'envs') else env.unwrapped.spec.id
    #print(env_id,'Reset')
    if env_id in ATARI_ENVS:
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

def sample_next_obs(env,state,action,state_prev=None,action_prev=None,n_avg=1):
    '''returns mean next obs over n_avg (sp~P(sp|s,a)) single step simulations'''
    obs_nexts = []
    for i in range(n_avg):
        obs = reset_env(env,state,state_prev,action_prev) #ATARI requires prev state/action for single step roll-in
        obs_nexts.append(env.step(action)[0])
    return np.mean(obs_nexts,axis=0)



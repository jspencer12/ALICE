import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.tf_layers import conv_to_fc,linear
from CIL_venv.lib.rl_baselines_zoo.utils import ALGOS, create_test_env, find_saved_model

EAGER = False
if EAGER:
    tf.compat.v1.enable_eager_execution()


def evaluate(env, model,model2=None,select=0):
    """Return mean fitness (sum of episodic rewards) for given model"""
    episode_rewards = []
    for _ in range(1):
        reward_sum = 0
        done = False
        obs = env.reset()
        t = 0
        while not done:
            env.render()
            if model is not None:
                action0 = [np.argmax(model(prepro(obs)))]
            else:
                action0 = 0
            if model2 is not None:
                action1, _states = model2.predict(obs)
                #print(model2.action_probability(obs))
            else:
                action1 = 1
            #print(tf.nn.softmax(model(prepro(obs))).numpy(),action0,action1)
            action = action1 if select else action0
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            #if t>10:
            #    done = True
            #t+= 1
        episode_rewards.append(reward_sum)
    return np.mean(episode_rewards)
def basic_eval(env,pi):
    episode_rewards = []
    for _ in range(1):
        reward_sum = 0
        done = False
        obs = env.reset()
        t = 0
        while not done:
            env.render()
            action = pi(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            #if t>10:
            #    done = True
            #t+= 1
        episode_rewards.append(reward_sum)
    return np.mean(episode_rewards)
def nature_cnn_keras(inputs,seed=None):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """

    initializer = tf.glorot_uniform_initializer(seed=seed,) 
    layer_1 = tf.keras.layers.Conv2D(filters=32, name='c1', kernel_size=8, strides=4, activation='relu',kernel_initializer=initializer)(inputs)
    #layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = tf.keras.layers.Conv2D(filters=64, name='c2', kernel_size=4, strides=2, activation='relu',kernel_initializer=initializer)(layer_1)
    #layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf.keras.layers.Conv2D(filters=64, name='c3', kernel_size=3, strides=1, activation='relu',kernel_initializer=initializer)(layer_2)
    #layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    out = tf.keras.layers.Dense(units=512, name='fc1', activation='relu')(layer_3)
    return out
def mlp_features(inputs,seed=None):
    #tanh activation
    pass
def prepro(obs):
    return tf.cast(obs/255.0,tf.float32)
def build_keras_cnn(env):
    inp = tf.cast(tf.keras.Input(shape=env.observation_space.shape),tf.float32)
    feat_out = nature_cnn2(inp)
    pi_out = tf.keras.layers.Dense(units=action_dim(env.action_space), name='pi')(feat_out)
    
def build_tf_cnn(env,sess):
    inp = tf.cast(tf.keras.Input(shape=env.observation_space.shape),tf.float32)
    feat_out = nature_cnn(inp)
    
    pi_out = linear(feat_out, 'pi', action_dim(env.action_space), init_scale=0.01, init_bias=0)
    pi_out = tf.keras.layers.Dense(units=action_dim(env.action_space), name='pi')(feat_out)
    #sess.run(tf.global_variables_initializer())
    return tf.keras.Model(inputs=inp,outputs=pi_out)
def action_dim(action_space):
    if type(action_space) is gym.spaces.discrete.Discrete:
        return action_space.n
    elif type(action_space) is gym.spaces.box.Box:
        if len(action_space.shape)==1:
            return action_space.shape[0]
        return action_space.shape
    else:
        raise NotImplementedError('Not yet supporting action space of type: {}'.format(type(action_space)))
def load_sb_expert(env_id,algo):
    log_path = 'CIL_venv/lib/rl_baselines_zoo/trained_agents/'+algo
    model_path = find_saved_model(algo, log_path, env_id, load_best=False)
    model = ALGOS[algo].load(model_path, env=env)
    pi = lambda obs: model.predict(obs)[0]
    return pi

# Create env
#env = gym.make('CartPole-v1')
# Create policy with a small network
#model = A2C('MlpPolicy', env, ent_coef=0.0, learning_rate=0.1,
#            policy_kwargs={'net_arch': [8, ]})

# Use traditional actor-critic policy gradient updates to
# find good initial parameters
#model.learn(total_timesteps=5000)

env_id = ['Acrobot-v1','CartPole-v1','BeamRiderNoFrameskip-v4','BreakoutNoFrameskip-v4','EnduroNoFrameskip-v4','PongNoFrameskip-v4','QbertNoFrameskip-v4','SeaquestNoFrameskip-v4','SpaceInvadersNoFrameskip-v4','MsPacmanNoFrameskip-v4','BipedalWalker-v2','LunarLander-v2','LunarLanderContinuous-v2','BipedalWalkerHardcore-v2'][0]
algo = ['A2C','ACER','ACKTR','PPO2','DQN'][3]
algo = algo.lower()
#model_path = find_saved_model(algo, log_path, env_id, load_best=False)
#env = make_atari_env(env_id, num_env=1, seed=0)
env = create_test_env(env_id, is_atari=False, should_render=True,log_dir=None,)


print('Action space shape',action_dim(env.action_space))

if 1:
    pi = load_sb_expert(env_id,algo)
    basic_eval(env,pi)

if 0:
    model = ALGOS[algo].load(model_path, env=env)
    mean_params = model.get_parameters()
    mean_params = dict((key,value) for key, value in mean_params.items() if ('c' in key or 'pi' in key) )
    weight_list = [np.squeeze(value) for key, value in mean_params.items()]
    #print([w.shape for w in weight_list])
    with tf.Session() as sess:
        model2 = build_cnn(env,sess)
    #print([w.shape for w in model2.get_weights()])
    model2.set_weights(weight_list)
    model2 = build_cnn(env)
    print([w.shape for w in model2.get_weights()])
    #evaluate(env,model2,model,select=0)
    outputs = list(map(lambda tname: tf.get_default_graph().get_tensor_by_name(tname), [
    'fc1/bias:0',
    ]))
    with tf.Session() as sess:
        val_outputs = sess.run(outputs)
        print(val_outputs)
if 0:
    model = ALGOS[algo].load(model_path, env=env)
    mean_params = model.get_parameters()
    mean_params = dict((key,value) for key, value in mean_params.items() if ('c' in key or 'pi' in key) )
    weight_list = [np.squeeze(value) for key, value in mean_params.items()]
    #print([w.shape for w in weight_list])
    with tf.Session() as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        model2 = build_cnn(env,sess)
        print([w.shape for w in model2.get_weights()])
        print(model2.get_config())
        print(tf.get_default_graph().get_collection('variables'))
        print(tf.get_default_graph().get_collection('trainable_variables'))
        model2.set_weights(weight_list)
        #print(model2.get_weights()[-1][0],weight_list[-1][0])
        print(mean_params.keys())
        evaluate(env,None,model,select=1)
        print(tf.get_default_graph().get_collection('variables'))
        print(tf.get_default_graph().get_collection('trainable_variables'))
        outputs = list(map(lambda tname: tf.get_default_graph().get_tensor_by_name(tname), [
        'fc1/bias:0',
        ]))
        val_outputs = sess.run(outputs)
        print(val_outputs)
    #print(dict((key,value.shape) for key, value in mean_params.items()))
    #evaluate(env,model,js=False)
    #evaluate(env,model2,model,select=0)

        

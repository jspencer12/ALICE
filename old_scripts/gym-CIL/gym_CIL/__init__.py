from gym.envs.registration import register

register(
    id='LQR-v0',
    entry_point='gym_CIL.envs:LQREnv',
)

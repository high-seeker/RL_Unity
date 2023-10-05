from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

env_name = "./push-block"
env = UnityToGymWrapper(unity_env=UnityEnvironment(file_name=env_name))
state = env.reset()
next_state, reward, done, info = env.step([0])
env.close()

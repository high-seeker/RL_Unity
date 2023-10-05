from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import DQN

# Initialize the Unity environment
unity_env = UnityEnvironment(file_name="./push-block", no_graphics=False)
gym_env = UnityToGymWrapper(unity_env)

# Initialize the model and train
model = DQN("MlpPolicy", gym_env, verbose=1)

model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_unity_agent")

# Close the environments
gym_env.close()
unity_env.close()
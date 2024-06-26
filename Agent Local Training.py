import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import torch

# Check if GPU can be used via CUDA
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create the log directories for both algorithms to be used with tensorboard
dqn_log_dir = "./dqn_frozenlake_tensorboard/"
ppo_log_dir = "./ppo_frozenlake_tensorboard/"
os.makedirs(dqn_log_dir, exist_ok=True)
os.makedirs(ppo_log_dir, exist_ok=True)

# Make the FrozenLake environment and wrap it with Monitor
env = gym.make('FrozenLake-v1', map_name='4x4')
env = Monitor(env)

# Check the observation and action space
n_obs = env.observation_space
n_actions = env.action_space

print(f"Number of observations: {n_obs}")
print(f"Number of actions: {n_actions}")

# Observation and action space overview
obs_space_overview = f"Observation Space: {n_obs}\nAction Space: {n_actions}\n"
reward_logic_overview = (
    "Reward Logic:\n"
    " - Goal (G): 1 reward for reaching the goal\n"
    " - Frozen (F): 0 reward for stepping onto a frozen lake\n"
)

# DQN algorithm training
dqn_model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.0001, buffer_size=50000, exploration_fraction=0.2, tensorboard_log=dqn_log_dir, device=device)
dqn_model.learn(total_timesteps=100000, progress_bar=True)
dqn_model.save('dqn_frozenlake')

mean_reward_dqn, std_reward_dqn = evaluate_policy(dqn_model, env, n_eval_episodes=20)
print(f'DQN - Mean reward: {mean_reward_dqn} +/- {std_reward_dqn}')

# PPO algorithm training
ppo_model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001, n_steps=4096, batch_size=256, n_epochs=20, tensorboard_log=ppo_log_dir, device=device)
ppo_model.learn(total_timesteps=100000)
ppo_model.save('ppo_frozenlake')

mean_reward_ppo, std_reward_ppo = evaluate_policy(ppo_model, env, n_eval_episodes=20)
print(f'PPO - Mean reward: {mean_reward_ppo} +/- {std_reward_ppo}')

# Run the DQN algoritm for many episodes and print the results from each episode and the mean reward
dqn_results = []
for episode in range(20):
    env_final, _ = env.reset()
    done = False
    truncated = False
    total_rewards = []
    while not done and not truncated:
        action, _states = dqn_model.predict(env_final, deterministic=True)
        action = int(action)  # Ensure action is an integer
        env_final, reward, done, truncated, info = env.step(action)
        env.render()  # Render the current state of the environment
        total_rewards.append(reward)
    episode_result = f"Episode {episode + 1} finished with total reward: {sum(total_rewards)} and rewards per step: {total_rewards}"
    print(episode_result)
    dqn_results.append(episode_result)

# Run the PPO algorithm for many episodes and print the results from each episode and the mean reward
ppo_results = []
for episode in range(20):
    env_final, _ = env.reset()
    done = False
    truncated = False
    total_rewards = []
    while not done and not truncated:
        action, _states = ppo_model.predict(env_final, deterministic=True)
        action = int(action)
        env_final, reward, done, truncated, info = env.step(action)
        env.render()
        total_rewards.append(reward)
    episode_result = f"Episode {episode + 1} finished with total reward: {sum(total_rewards)} and rewards per step: {total_rewards}"
    print(episode_result)
    ppo_results.append(episode_result)

# Write the results to a text file to keep it organized and deliver the results
with open("results.txt", "w") as file:
    file.write("DQN Results:\n")
    file.write(f"Mean reward: {mean_reward_dqn} +/- {std_reward_dqn}\n")
    for result in dqn_results:
        file.write(result + "\n")

    file.write("\nPPO Results:\n")
    file.write(f"Mean reward: {mean_reward_ppo} +/- {std_reward_ppo}\n")
    for result in ppo_results:
        file.write(result + "\n")

    file.write("\nObservation Space and Action Space Overview:\n")
    file.write(obs_space_overview)

    file.write("\nAgent's Reward Logic:\n")
    file.write(reward_logic_overview)

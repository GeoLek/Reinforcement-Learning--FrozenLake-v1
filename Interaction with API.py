import requests
import numpy as np


# Function to interact with the API and evaluate the agent
def evaluate_agent(model_type, num_episodes=100):
    rewards = []
    for episode in range(num_episodes):
        try:
            # Start a new game
            response = requests.post('http://localhost:5005/new_game')
            response.raise_for_status()
            env_id = response.json().get('env_id')
            print(f"New game started, env_id: {env_id}")
            # Reset the environment
            response = requests.post('http://localhost:5005/reset', json={'env_id': env_id})
            response.raise_for_status()
            obs = np.array(response.json().get('observation'))
            print(f"Environment reset, initial observation: {obs}")
            done = False
            total_reward = 0
            while not done:
                # Predict the next action
                response = requests.post('http://localhost:5005/predict', json={
                    'env_id': env_id,
                    'model_type': model_type,
                    'observation': obs.tolist()
                })
                response.raise_for_status()
                action = response.json().get('action')
                print(f"Predicted action: {action}")
                # Take the action in the environment
                response = requests.post('http://localhost:5005/step', json={
                    'env_id': env_id,
                    'action': action
                })
                response.raise_for_status()
                step_data = response.json()
                obs = np.array(step_data.get('observation'))
                reward = step_data.get('reward')
                done = step_data.get('done')
                print(f"Step taken, observation: {obs}, reward: {reward}, done: {done}")
                total_reward += reward
            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes} finished with total reward: {total_reward}")
        except requests.exceptions.RequestException as e:
            print(f"Error during episode {episode + 1}: {e}")
            break
    return np.mean(rewards), np.std(rewards), rewards
# Evaluate the DQN agent
mean_reward_dqn, std_reward_dqn, dqn_rewards = evaluate_agent('dqn')
print(f'DQN - Mean reward: {mean_reward_dqn} +/- {std_reward_dqn}')
# Evaluate the PPO agent
mean_reward_ppo, std_reward_ppo, ppo_rewards = evaluate_agent('ppo')
print(f'PPO - Mean reward: {mean_reward_ppo} +/- {std_reward_ppo}')
# Save results to a file
with open('evaluation_results.txt', 'w') as f:
    f.write('DQN Results:\n')
    f.write(f'Mean reward: {mean_reward_dqn} +/- {std_reward_dqn}\n')
    f.write('Rewards per episode:\n')
    f.write('\n'.join(map(str, dqn_rewards)))
    f.write('\n\nPPO Results:\n')
    f.write(f'Mean reward: {mean_reward_ppo} +/- {std_reward_ppo}\n')
    f.write('Rewards per episode:\n')
    f.write('\n'.join(map(str, ppo_rewards)))
print('Evaluation results saved to evaluation_results.txt')
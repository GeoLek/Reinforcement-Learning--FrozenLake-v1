from flask import Flask, jsonify, request
import gymnasium as gym
import uuid
from uuid import uuid4
from typing import Dict, Any
from stable_baselines3 import DQN, PPO

class FrozenLakeAPI:
    def __init__(self) -> None:
        self.app: Flask = Flask(__name__)
        self.app.debug = True
        self.games: Dict[str, gym.Env] = {}
        self.dqn_model = DQN.load('dqn_frozenlake')
        self.ppo_model = PPO.load('ppo_frozenlake')

    def run_server(self) -> None:
        self.app.route('/new_game', methods=['POST'])(self.new_game)
        self.app.route('/step', methods=['POST'])(self.step)
        self.app.route('/reset', methods=['POST'])(self.reset)
        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.run(host="localhost", port=5005, threaded=True)

    def new_game(self) -> Any:
        env_id = str(uuid.uuid4())
        env = gym.make('FrozenLake-v1', map_name='4x4')
        self.games[env_id] = env
        return jsonify({'env_id': env_id})

    def reset(self) -> Any:
        data = request.json
        env_id = data.get('env_id')
        if not env_id or env_id not in self.games:
            return jsonify({'error': 'Invalid environment ID'}), 400
        try:
            obs, _ = self.games[env_id].reset()
            if isinstance(obs, (int, float)):
                obs = [obs]
            else:
                obs = obs.tolist()
            return jsonify({'observation': obs})
        except Exception as e:
            print(f"Error during reset: {e}")
            return jsonify({'error': str(e)}), 500

    def step(self) -> Any:
        data = request.json
        env_id = data.get('env_id')
        action = data.get('action')
        if not env_id or env_id not in self.games:
            return jsonify({'error': 'Invalid environment ID'}), 400
        try:
            env = self.games[env_id]
            obs, reward, done, truncated, info = env.step(action)
            if isinstance(obs, (int, float)):
                obs = [obs]
            else:
                obs = obs.tolist()
            return jsonify({
                'observation': obs,
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
            })
        except Exception as e:
            print(f"Error during step: {e}")
            return jsonify({'error': str(e)}), 500

    def predict(self) -> Any:
        data = request.json
        env_id = data.get('env_id')
        model_type = data.get('model_type')
        obs = data.get('observation')

        if not env_id or env_id not in self.games:
            return jsonify({'error': 'Invalid environment ID'}), 400
        try:
            if model_type == 'dqn':
                action, _states = self.dqn_model.predict(obs, deterministic=True)
            elif model_type == 'ppo':
                action, _states = self.ppo_model.predict(obs, deterministic=True)
            else:
                return jsonify({'error': 'Invalid model type'}), 400

            return jsonify({'action': int(action)})
        except Exception as e:
            print(f"Error during predict: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    emulation_api = FrozenLakeAPI()
    emulation_api.run_server()
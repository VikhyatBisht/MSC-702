import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
import pandas as pd
import highway_env
import numpy as np

# ACTIONS mapping
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

TRAIN = False  # Set to False since we're focusing on data collection

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Load the trained model
    model = A2C.load("highway_a2c/model", env=env)

    # Lists to store data
    observations = []
    actions_taken = []
    rewards_received = []
    infos = []

    # Run the trained model and record video
    env = RecordVideo(env, video_folder="highway_a2c/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})

    for videos in range(1000):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=False)
            
            # Convert action to integer if it's a numpy array
            if isinstance(action, np.ndarray):
                action = action.item()
            
            # Convert action to string representation and store
            action_str = ACTIONS_ALL[action]
            actions_taken.append(action_str)
            
            # Convert observation to string representation and store
            obs_str = str(obs.tolist())
            observations.append(obs_str)
            
            # Step in the environment
            results = env.step(action)
            obs = results[0]
            reward = results[1]
            done = results[2]
            info = results[3]
            
            # Store more data
            rewards_received.append(reward)
            infos.append(info)
            
            # Render
            env.render()

    env.close()

    # Store collected data in a DataFrame and save to CSV
    df = pd.DataFrame({
        'Observations': observations,
        'Actions': actions_taken,
        'Rewards': rewards_received,
        'Info': infos
    })

    df.to_csv('collected_data_a2c.csv', index=False)

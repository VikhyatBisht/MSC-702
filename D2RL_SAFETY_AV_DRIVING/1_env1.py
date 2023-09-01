import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C

import highway_env

TRAIN = True

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model using A2C
    model = A2C('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                gamma=0.8,
                n_steps=5,  # Number of steps for each rollout
                ent_coef=0.01,  # Entropy coefficient for exploration
                verbose=1,
                tensorboard_log="highway_a2c/")  # Change log directory

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_a2c/model")  # Change save directory
        del model

    # Run the trained model and record video
    model = A2C.load("highway_a2c/model", env=env)  # Change load directory
    env = RecordVideo(env, video_folder="highway_a2c/videos", episode_trigger=lambda e: True)  # Change video directory
    env.configure({"simulation_frequency": 5})  # Setting simulation frequency as per new information

    for videos in range(1000):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()

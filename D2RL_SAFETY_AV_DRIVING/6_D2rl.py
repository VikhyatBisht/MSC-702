import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
import pandas as pd
from tensorflow import keras

class BehaviorCloningPolicy:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, ego_vehicle, vehicles):
        obs = [self.vehicle_to_observation(ego_vehicle)]
        for vehicle in vehicles:
            obs.append(self.vehicle_to_observation(vehicle))

        while len(obs) < 5:
            obs.append([0, 0, 0, 0, 0])  # Padding

        obs = np.array(obs).reshape(1, 5, 5)
        action_probs = self.model.predict(obs)
        return np.argmax(action_probs[0])

    def vehicle_to_observation(self, vehicle):
        return [
            1.0,
            vehicle.position[0] / 100.0,
            vehicle.position[1] / 5.0,
            vehicle.speed / 30.0,
            vehicle.heading / (2 * np.pi)
        ]

class D2RLNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(D2RLNetwork, self).__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), features_dim),
            nn.ReLU()
        )
        self.d2rl1 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        self.d2rl2 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.flatten(observations)
        x1 = self.hidden(x)
        x2 = self.d2rl1(x1)
        x3 = self.d2rl2(x1 + x2)
        return x3

class D2RLPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(D2RLPolicy, self).__init__(*args, **kwargs, features_extractor_class=D2RLNetwork, features_extractor_kwargs=dict(features_dim=256))

class CustomHighwayEnv(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info,_ = self.env.step(action)
        if done:
            reward = +100
        else:
            reward = -5
        return obs, reward, done, info,_  

if __name__ == "__main__":
    # Create environment
    base_env = gym.make("highway-fast-v0")
    env = CustomHighwayEnv(gym.make("highway-fast-v0" , render_mode="rgb_array"))

    # Load the pretrained behavior cloning model
    behavior_model = keras.models.load_model("behavior_cloning_model.h5")
    bc_policy = BehaviorCloningPolicy("behavior_cloning_model.h5")

    # Train A2C with the custom D2RL policy
    model = A2C(D2RLPolicy, env, verbose=1)
    for _ in range(10):  # Train for 10 epochs
        model.learn(total_timesteps=100)
        
        # Render one episode after each epoch for visualization
        
        obs = env.reset()
        if len(obs) > 1:
            obs = obs[0]
        done = False
        while not done:
            
            action, _states = model.predict(obs, deterministic=True)
            obs, _, done, _,_ = env.step(action)
            env.render()

    # Save the trained model
    model.save("a2c_d2rl_model")

    # Test the trained model and save rewards to CSV
    model = A2C.load("a2c_d2rl_model", env=env)

    episode_data = {}
    for ep in range(10):
        obs = env.reset()
        
        if len(obs) > 1:
            obs = obs[0]
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _,_ = env.step(action)
            episode_reward += reward
            
            env.render()  # Render each step during testing
        episode_data[ep+1]=episode_reward

    # Close the rendering window after testing
    env.close()

    # Save episode rewards to CSV
    pd.DataFrame({"Episode ": episode_data.keys(),"Reward" :episode_data.values() }).to_csv("d2rl_episode_rewards1.csv", index=False)
    print("Training and testing complete!")

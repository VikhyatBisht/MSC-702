import gymnasium as gym
import highway_env
import numpy as np
import tensorflow as tf
from tensorflow import keras


class BehaviorCloningPolicy:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, ego_vehicle, vehicles):
        # Derive observation relative to ego vehicle
        obs = [self.vehicle_to_observation(ego_vehicle)]
        for vehicle in vehicles:
            obs.append(self.vehicle_to_observation(vehicle))

        while len(obs) < 5:
            obs.append([0, 0, 0, 0, 0])  # Padding with "absence" observation

        obs = np.array(obs).reshape(1, 5, 5)

        # Predict action using behavior cloning model
        action_probs = self.model.predict(obs)
        return np.argmax(action_probs[0])

    def vehicle_to_observation(self, vehicle):
        return [
            1.0,  # Presence flag
            vehicle.position[0] / 100.0,  # Max road length approximation
            vehicle.position[1] / 5.0,    # Max road width approximation
            vehicle.speed / 30.0,         # Max speed approximation
            vehicle.heading / (2 * np.pi)
        ]


def simulate_with_behavior_cloning_agents():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    bc_policy = BehaviorCloningPolicy('behavior_cloning_model.h5')

    for _ in range(1000):
        done = False
        obs = env.reset()
        while not done:
            ego_vehicle = env.vehicle
            # Choose four random vehicles as the other vehicles
            other_vehicles = list(np.random.choice(env.road.vehicles, size=4, replace=False))

            agent_action = bc_policy.predict(ego_vehicle, other_vehicles)
            obs, reward, done, _, _ = env.step(agent_action)
            env.render()


if __name__ == "__main__":
    simulate_with_behavior_cloning_agents()

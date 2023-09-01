import os
import gymnasium as gym
import highway_env
import numpy as np
import tensorflow as tf
from tensorflow import keras

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

class BehaviorCloningPolicy:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, ego_vehicle, vehicles):
        obs = [self.vehicle_to_observation(ego_vehicle)]
        for vehicle in vehicles:
            obs.append(self.vehicle_to_observation(vehicle))
        while len(obs) < 5:
            obs.append([0, 0, 0, 0, 0])  # Padding with "absence" observation
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

def perturb_and_save_model(model, perturbation_factor=0.05, number_of_models=5):
    # Create directory to save models if it doesn't exist
    if not os.path.exists("perturbed_models"):
        os.makedirs("perturbed_models")

    model_paths = []
    for i in range(number_of_models):
        perturbed_model = keras.models.clone_model(model)
        perturbed_model.set_weights(model.get_weights())
        
        # Apply perturbation
        for layer in perturbed_model.layers:
            weights = layer.get_weights()
            perturbed_weights = [w + np.random.normal(0, perturbation_factor, size=w.shape) for w in weights]
            layer.set_weights(perturbed_weights)

        # Save the perturbed model
        path = os.path.join("perturbed_models", "perturbed_model_{}.h5".format(i))
        perturbed_model.save(path)
        model_paths.append(path)

    return model_paths

def simulate_with_enhanced_agents(model_paths):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    agents = [BehaviorCloningPolicy(model_path) for model_path in model_paths]
    
    for _ in range(1000):
        done = False
        obs = env.reset()
        while not done:
            ego_vehicle = env.vehicle
            other_vehicles = list(np.random.choice(env.road.vehicles, size=4, replace=False))

            # Pick a random agent from our list of trained agents
            agent = np.random.choice(agents)
            agent_action = agent.predict(ego_vehicle, other_vehicles)
            obs, reward, done, _, _ = env.step(agent_action)
            env.render()

if __name__ == "__main__":
    original_model_path = "behavior_cloning_model.h5"
    original_model = keras.models.load_model(original_model_path)
    model_paths = perturb_and_save_model(original_model)
    simulate_with_enhanced_agents(model_paths)

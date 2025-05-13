from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(env, timesteps=10000):
    env = DummyVecEnv([lambda: env])  # Wrap in vectorized env
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_agent(model, env):
    env = DummyVecEnv([lambda: env])
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    return total_reward

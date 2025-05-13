import gymnasium as gym

from stable_baselines3 import DDPG

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v3', render_mode=None)
    env.reset()

    model = DDPG('MlpPolicy', env)
    model.learn(total_timesteps=10000)

    total_reward = 0
    total_steps = 0

    for iteration in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)

            new_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            total_steps += 1
            state = new_state

            # env.render()

        if iteration == 49:
            print(
                f"10000 Episodes, 50 Iterations: average reward {total_reward / 50}, average steps {total_steps / 50}")

    print(f"10000 Episodes, 100 Iterations: average reward {total_reward / 100}, average steps {total_steps / 100}")

    model = DDPG('MlpPolicy', env)
    model.learn(total_timesteps=50000)

    total_reward = 0
    total_steps = 0

    for iteration in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)

            new_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            total_steps += 1
            state = new_state

            # env.render()

        if iteration == 49:
            print(
                f"50000 Episodes, 50 Iterations: average reward {total_reward / 50}, average steps {total_steps / 50}")

    print(f"50000 Episodes, 100 Iterations: average reward {total_reward / 100}, average steps {total_steps / 100}")

import gymnasium as gym
import torch

from deep_q_learning_torch import DDPG, OrnsteinUhlenbeckActionNoise

if __name__ == '__main__':
    device = 'cuda'
    env = gym.make('LunarLanderContinuous-v2', render_mode=None)
    env.reset()

    agent = DDPG(state_space_shape=(8,), action_space_shape=(2,),
                 learning_rate_actor=0.001, learning_rate_critic=0.001,
                 discount_factor=0.99, batch_size=32, memory_size=10000).to(device)

    num_episodes = 5000
    num_steps_per_episode = 1000

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape=(2,))

    train_reward = 0
    train_steps = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        noise.reset()

        for step in range(num_steps_per_episode):
            if episode % 5 == 0:
                action = agent.get_action(state, discrete=False) + noise()
            else:
                action = agent.get_action(state, discrete=False)
            action = 2 * action - 1
            action = action.cpu().numpy()

            new_state, reward, terminated, _, _ = env.step(action)

            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            agent.update_memory(state, action, reward, new_state, terminated)

            train_reward += reward
            train_steps += 1

            if terminated:
                break

            state = new_state

        agent.train()

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

    print(f"5000 Episodes, Average train reward: {train_reward / 5000}, average steps {train_steps / 5000}")

    total_reward = 0
    total_steps = 0

    for iteration in range(100):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < num_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.actor(state_tensor).cpu().numpy()

            action = 2 * action - 1

            new_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            total_steps += 1
            steps += 1
            state = new_state

        if iteration == 49:
            print(f"5000 Episodes, 50 Iterations: average reward {total_reward/50}, average steps {total_steps/50}")

    print(f"5000 Episodes, 100 Iterations: average reward {total_reward / 100}, average steps {total_steps / 100}")



    num_episodes = 20000

    train_reward = 0
    train_steps = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        noise.reset()

        for step in range(num_steps_per_episode):
            if episode % 5 == 0:
                action = agent.get_action(state, discrete=False) + noise()
            else:
                action = agent.get_action(state, discrete=False)
            action = 2 * action - 1
            action = action.cpu().numpy()

            new_state, reward, terminated, _, _ = env.step(action)

            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            agent.update_memory(state, action, reward, new_state, terminated)

            train_reward += reward
            train_steps += 1

            if terminated:
                break

            state = new_state

        agent.train()

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

    print(f"20000 Episodes, Average train reward: {train_reward / 20000}, average steps {train_steps / 20000}")

    total_reward = 0
    total_steps = 0

    for iteration in range(100):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < num_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.actor(state_tensor).cpu().numpy()

            action = 2 * action - 1

            new_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            total_steps += 1
            steps += 1
            state = new_state

        if iteration == 49:
            print(f"20000 Episodes, 50 Iterations: average reward {total_reward/50}, average steps {total_steps/50}")

    print(f"20000 Episodes, 100 Iterations: average reward {total_reward / 100}, average steps {total_steps / 100}")

import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Rate at which exploration rate decays
episodes = 10000  # Total number of episodes
max_steps = 150  # Max steps per episode

# Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# Q-learning algorithm
def q_learning(env, num_episodes, learning_rate, discount_factor, epsilon_decay):
    # Initialize Q-table with zeros
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    Q = np.zeros((state_space_size, action_space_size))

    # Initialize exploration parameters
    epsilon = 1.0  # Initial exploration rate
    min_epsilon = 0.01  # Minimum exploration rate
    decay_rate = epsilon_decay  # Epsilon decay rate

    # Store rewards for each episode
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment for a new episode and extract state
        done = False
        total_reward = 0

        while not done:
            # Ensure the state is an integer
            state = int(state)

            # Choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(Q[state])  # Exploit learned values

            # Take action and observe new state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Update done flag

            # Ensure the next_state is also an integer
            next_state = int(next_state)

            # Update Q-table using the Q-learning formula
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += learning_rate * (
                    reward + discount_factor * Q[next_state, best_next_action] - Q[state, action]
            )

            # Update the state and total reward
            state = next_state
            total_reward += reward

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= decay_rate

        # Store the total reward for this episode
        rewards.append(total_reward)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return rewards, Q


def evaluate_agent(q_table):
    total_rewards = []
    for episode in range(100):
        state = env.reset()  # Reset the environment
        total_reward = 0
        done = False  # Initialize done flag

        # If the state is a tuple, take the first element
        if isinstance(state, tuple):
            state = state[0]  # or handle it based on your environment

        for _ in range(max_steps):
            action = np.argmax(q_table[state])  # Always choose the best action

            # Unpack the returned values; adjust this based on your environment's output
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, done, info, extra_info = step_output  # Unpack all 5 outputs
            elif len(step_output) == 4:
                next_state, reward, done, info = step_output
            elif len(step_output) == 3:
                next_state, reward, done = step_output
                info = {}
            else:
                raise ValueError(f"Unexpected number of outputs from env.step: {len(step_output)}")

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            total_reward += reward
            state = next_state  # Update the state

            if done:
                break

        total_rewards.append(total_reward)

    print(f"Average reward over 100 episodes: {np.mean(total_rewards)}")


# Plotting cumulative rewards over episodes
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Cumulative Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Training the agent...")

    # Call q_learning with the required arguments
    rewards, q_table = q_learning(env, episodes, alpha, gamma, epsilon_decay)

    print("Evaluating the trained agent...")
    evaluate_agent(q_table)

    print("Plotting results...")
    plot_rewards(rewards)

    print("Training completed.")


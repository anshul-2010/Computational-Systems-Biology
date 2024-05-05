"""
Incomplete attempt to incorporate the RL agent for gene regulation
""" 
import numpy as np

all_states = [i for i in range(2**3)]
print(np.array(all_states, dtype=np.uint8))

class Boolean_GRN_Environment:
  """
  The Gene Regulatory Network was converted into a Boolean network and the different possible variations, i.e., 
  the binary strings are the nodes of this new graph network. This forms the Environment.

  Args:
  """

  def __init__(self,
               name:str,
               string_length:int,
               ):

    self.name = name
    self.string_length = string_length
    self.state_space = self._get_state_space()
    # self.agents = agents
    # self.num_agents = len(agents)
    self.reset()

  def _get_state_space(self):
    """
    Defines the state space as all possible binary strings.

    Returns:
      A NumPy array representing the state space.
    """
    all_states = [i for i in range(2**self.string_length)]
    return np.array(all_states, dtype=np.uint8)

  def reset(self):
    """
    Resets the environment to a random starting node.

    Returns:
      The initial state as a binary string.
    """
    self.current_state = np.random.choice(self.state_space)
    return self.current_state

  def step(self, action):
    """
    Takes an action (mutation position) and transitions to a new state.

    Args:
      action: An integer representing the bit position to flip.

    Returns:
      A tuple containing the new state, reward, done flag, and info.
    """

    # Validate action within string length
    if action < 0 or action >= self.string_length:
        raise ValueError("Invalid action: Out of bounds")

    new_state = self.current_state.copy()
    # Flip the bit using bitwise XOR
    new_state ^= (1 << action)  # Equivalent to new_state[action] = 1 - new_state[action]
    self.current_state = new_state

    # Define your reward function here (e.g., reward for getting closer to goal string)
    reward = np.sum(np.binary_repr(new_state, width=self.string_length).count('1'))

    # Done flag (optional, can be used for termination conditions)
    done = False

    # Info dictionary with original state and number of 1s in both states
    info = {
      "original_state": self.current_state.tobytes().decode('utf-8'),
      "original_ones": np.sum(self.current_state),
      "new_state": new_state.tobytes().decode('utf-8'),
      "new_ones": reward
    }

    return new_state, reward, done, info

# Example usage
env = Boolean_GRN_Environment(name = "Hello",string_length = 5)
state = env.reset()

while True:
  # Choose an action (e.g., random bit position)
  action = np.random.randint(0, env.string_length)
  new_state, reward, done, info = env.step(action)
  print(np.binary_repr(new_state))
  break

class Agent:
  """
  Q-learning Agent for the Boolean_GRN_Environment.
  """

  def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9):
    """
    Args:
      state_space: A NumPy array representing the state space.
      action_space_size: Number of possible actions (length of the binary string).
      learning_rate: Learning rate for Q-learning (default: 0.1).
      discount_factor: Discount factor for Q-learning (default: 0.9).
    """
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

    # Initialize Q-table with zeros
    self.Q_table = np.zeros((self.state_space_size, self.action_space_size))

  def choose_action(self, state, epsilon=0.2):
    """
    Chooses an action based on the Q-table and epsilon-greedy exploration.

    Args:
      state: The current state as a binary string (index in state_space).
      epsilon: Exploration rate (default: 0.1).

    Returns:
      An integer representing the chosen action (bit position to flip).
    """
    # Exploit (choose action with highest Q-value)
    if np.random.rand() > epsilon:
      return np.argmax(self.Q_table[state, :])
    # Explore (random action)
    else:
      return np.random.randint(0, self.action_space_size)

  def learn(self, state, action, reward, next_state):
    """
    Updates the Q-table based on the Bellman equation for Q-learning.

    Args:
      state: The current state (index in state_space).
      action: The action taken.
      reward: The reward received.
      next_state: The next state (index in state_space).
    """
    # Get the maximum Q-value of the next state (exploit the future)
    max_Q = np.max(self.Q_table[next_state, :])

    # Q-learning update rule (Bellman equation)
    self.Q_table[state, action] += self.learning_rate * (reward + self.discount_factor * max_Q - self.Q_table[state, action])

# Define training parameters
string_length = 5  # Length of the binary string
episodes = 10  # Number of training episodes
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration rate

# Create the environment and agent
env = Boolean_GRN_Environment("Hello",string_length)
agent = Agent(env.state_space.size, string_length, learning_rate, discount_factor)

# Training loop
for episode in range(episodes):
  # Reset the environment
  state = env.reset()

  count = 0
  # Run the episode
  while True:
    # Choose an action based on epsilon-greedy exploration
    action = agent.choose_action(state, epsilon)

    # Take action, observe reward, next state, and done flag
    next_state, reward, done, info = env.step(action)

    # Update the Q-table based on the experience
    agent.learn(state, action, reward, next_state)

    # Update state for next action
    state = next_state
    print(np.binary_repr(state), state)
    count += 1

    # Terminate episode if done
    if done or count == 10:
      break

  # Print episode stats (optional)
  print(f"Episode: {episode+1}, Reward: {info['new_ones']}")

# Test the agent on a new state (optional)
new_state = env.reset()
action = agent.choose_action(new_state, 0.0)  # Exploit (no exploration)
next_state, reward, done, info = env.step(action)
print(f"\nFinal State: {info['new_state']}, Number of 1s: {info['new_ones']}")


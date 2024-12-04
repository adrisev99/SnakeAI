import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Snake Game Environment
class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.block_size = 20
        self.reset()
        
    def reset(self):
        # Initialize snake at the center
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (self.block_size, 0)  # Start moving right
        self.food = self._place_food()
        self.score = 0
        self.steps_without_food = 0
        return self._get_state()
    
    def _place_food(self):
        while True:
            x = random.randrange(0, self.width, self.block_size)
            y = random.randrange(0, self.height, self.block_size)
            food = (x, y)
            if food not in self.snake:
                return food
    
    def _get_state(self):
        # Construct game state for AI
        head = self.snake[0]
        
        # Detect danger directions
        left_danger = head[0] - self.block_size < 0 or (head[0] - self.block_size, head[1]) in self.snake
        right_danger = head[0] + self.block_size >= self.width or (head[0] + self.block_size, head[1]) in self.snake
        up_danger = head[1] - self.block_size < 0 or (head[0], head[1] - self.block_size) in self.snake
        down_danger = head[1] + self.block_size >= self.height or (head[0], head[1] + self.block_size) in self.snake
        
        # Food direction
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        # Current direction
        dir_left = self.direction == (-self.block_size, 0)
        dir_right = self.direction == (self.block_size, 0)
        dir_up = self.direction == (0, -self.block_size)
        dir_down = self.direction == (0, self.block_size)
        
        return [
            # Danger state
            left_danger, right_danger, up_danger, down_danger,
            
            # Current direction
            dir_left, dir_right, dir_up, dir_down,
            
            # Food direction
            food_left, food_right, food_up, food_down
        ]
    
    def step(self, action):
        # 0: straight, 1: right turn, 2: left turn
        current_head = self.snake[0]
        
        # Determine new direction based on action
        if action == 1:  # Right turn
            if self.direction == (self.block_size, 0):
                new_dir = (0, -self.block_size)
            elif self.direction == (-self.block_size, 0):
                new_dir = (0, self.block_size)
            elif self.direction == (0, self.block_size):
                new_dir = (self.block_size, 0)
            else:  # up
                new_dir = (-self.block_size, 0)
        elif action == 2:  # Left turn
            if self.direction == (self.block_size, 0):
                new_dir = (0, self.block_size)
            elif self.direction == (-self.block_size, 0):
                new_dir = (0, -self.block_size)
            elif self.direction == (0, self.block_size):
                new_dir = (-self.block_size, 0)
            else:  # up
                new_dir = (self.block_size, 0)
        else:  # Straight
            new_dir = self.direction
        
        # Move
        new_head = (current_head[0] + new_dir[0], current_head[1] + new_dir[1])
        
        # Check game over conditions
        game_over = False
        reward = 0
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            game_over = True
            reward = -10
        
        # Check self collision
        if new_head in self.snake:
            game_over = True
            reward = -10
        
        # Update snake
        self.snake.insert(0, new_head)
        self.direction = new_dir
        self.steps_without_food += 1
        
        # Check food eating
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
            self.steps_without_food = 0
        else:
            # Remove tail if not eating
            self.snake.pop()
        
        # Penalty for taking too long without eating
        if self.steps_without_food > 100:
            game_over = True
            reward = -10
        
        return self._get_state(), reward, game_over, self.score

# Deep Q-Network for Snake AI
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Reinforcement Learning Agent
class SnakeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_fn = nn.MSELoss()
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Replay memory
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax().item()
    
    def train(self):
        # Check if enough memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Training and Visualization
def train_snake_ai(episodes=1000, visualize_every=10, fps=30):
    # Initialize game and agent
    game = SnakeGame()
    state_size = len(game.reset())
    action_size = 3  # 0: straight, 1: right turn, 2: left turn
    agent = SnakeAgent(state_size, action_size)
    
    # Tracking
    scores = []
    
    # Pygame setup for visualization
    pygame.init()
    screen = pygame.display.set_mode((game.width, game.height))
    pygame.display.set_caption('Snake AI Training')
    clock = pygame.time.Clock()
    
    # Training loop
    for episode in range(episodes):
        # Reset game
        state = game.reset()
        done = False
        total_score = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, score = game.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            # Update state and score
            state = next_state
            total_score = score
            
            # Visualization
            if episode % visualize_every == 0:
                screen.fill((0, 0, 0))
                for segment in game.snake:
                    pygame.draw.rect(screen, (0, 255, 0), (*segment, game.block_size, game.block_size))
                pygame.draw.rect(screen, (255, 0, 0), (*game.food, game.block_size, game.block_size))
                pygame.display.flip()
                clock.tick(fps)  # Control game speed
        
        # Track scores
        scores.append(total_score)
        
        # Periodic updates
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_score}, Epsilon: {agent.epsilon:.2f}")
    
    # Plot learning progress
    plt.plot(scores)
    plt.title('Snake AI Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('snake_learning_progress.png')
    plt.close()
    
    pygame.quit()

# Visualization with Pygame
def visualize_snake_ai():
    pygame.init()
    game = SnakeGame()
    state_size = len(game.reset())
    action_size = 3
    agent = SnakeAgent(state_size, action_size)
    
    # Pygame setup
    screen = pygame.display.set_mode((game.width, game.height))
    pygame.display.set_caption('Snake AI')
    clock = pygame.time.Clock()
    
    # Game loop
    state = game.reset()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # AI selects action
        action = agent.select_action(state)
        
        # Take step
        next_state, reward, done, score = game.step(action)
        
        # Visualization
        screen.fill((0, 0, 0))
        
        # Draw snake
        for segment in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), 
                             (*segment, game.block_size, game.block_size))
        
        # Draw food
        pygame.draw.rect(screen, (255, 0, 0), 
                         (*game.food, game.block_size, game.block_size))
        
        pygame.display.flip()
        clock.tick(10)  # Control game speed
        
        # Update state
        state = next_state
    
    pygame.quit()

# Main execution
if __name__ == '__main__':
    # Train the AI
    train_snake_ai()
    
    # Visualize the trained AI
    visualize_snake_ai()

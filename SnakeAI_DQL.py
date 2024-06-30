import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pygame.math import Vector2
import matplotlib.pyplot as plt
import threading
import time
import os
import queue

# Neural Network for DQN
class DQNSnake(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNSnake, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SNAKE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(1, 0)
        self.new_block = False

    def move_snake(self):
        if self.new_block:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def update_direction(self, action):
        # Action: 0 = straight, 1 = right turn, 2 = left turn
        if action == 1:  # right turn
            if self.direction == Vector2(1, 0): self.direction = Vector2(0, 1)
            elif self.direction == Vector2(0, 1): self.direction = Vector2(-1, 0)
            elif self.direction == Vector2(-1, 0): self.direction = Vector2(0, -1)
            elif self.direction == Vector2(0, -1): self.direction = Vector2(1, 0)
        elif action == 2:  # left turn
            if self.direction == Vector2(1, 0): self.direction = Vector2(0, -1)
            elif self.direction == Vector2(0, 1): self.direction = Vector2(1, 0)
            elif self.direction == Vector2(-1, 0): self.direction = Vector2(0, 1)
            elif self.direction == Vector2(0, -1): self.direction = Vector2(-1, 0)

class FRUIT:
    def __init__(self):
        self.randomize()

    def randomize(self):
        self.x = random.randint(0, cell_number - 1)
        self.y = random.randint(0, cell_number - 1)
        self.pos = Vector2(self.x, self.y)

class ENVIRONMENT:
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.score = 0
        self.highest_score = 0

    def reset(self):
        self.snake.reset()
        self.fruit.randomize()
        self.score = 0
        return self.get_state()

    def step(self, action):
        self.snake.update_direction(action)
        self.snake.move_snake()

        reward = 0
        done = False

        if self.snake.body[0] == self.fruit.pos:
            self.fruit.randomize()
            self.snake.add_block()
            self.score += 1
            reward = 10  # Reward for eating fruit
        elif self.check_fail():
            done = True
            reward = -10  # Penalty for collision

        return self.get_state(), reward, done

    def get_state(self):
        head = self.snake.body[0]
        fruit = self.fruit.pos
        direction = self.snake.direction

        state = [head.x, head.y, fruit.x, fruit.y, direction.x, direction.y]
        return np.array(state)

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            return True
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                return True
        return False

    def render(self, episode, save_count):
        screen.fill((175, 215, 70))
        for block in self.snake.body:
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 0, 255), block_rect)

        fruit_rect = pygame.Rect(int(self.fruit.pos.x * cell_size), int(self.fruit.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(screen, (255, 0, 0), fruit_rect)

        font = pygame.font.Font(None, 36)
        text = font.render(f'Episode: {episode}, Saves: {save_count}, Highest Score: {self.highest_score}', True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.update()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNSnake(state_size, action_size)
        self.target_model = DQNSnake(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load_model(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded model from {filename}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.target_model(next_state)[0]).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Pygame setup
pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()

env = ENVIRONMENT()
agent = DQNAgent(state_size=6, action_size=3)  # 6 state parameters and 3 possible actions
batch_size = 32
episodes = 1000
save_interval = 100
save_count = 0

# Load model if exists
agent.load_model('dqn_snake.pth')

# Real-time plotting setup
scores = []
episodes_list = []
plot_queue = queue.Queue()

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(episodes_list, scores)
ax.set_xlim(0, episodes)
ax.set_ylim(0, 100)  # Adjust as needed for expected score range

ax.set_xlabel('Episode')
ax.set_ylabel('Score')

def update_plot():
    while not plot_queue.empty():
        episode, score = plot_queue.get()
        scores.append(score)
        episodes_list.append(episode)
    line.set_xdata(episodes_list)
    line.set_ydata(scores)
    ax.relim()
    ax.autoscale_view(True, True, True)  # Auto-adjust the axis
    plt.draw()
    plt.pause(0.01)

def train_agent():
    global scores, episodes_list, save_count
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            env.render(episode + 1, save_count)  # Update the game window
            clock.tick(30)  # Speed up the game to 30 frames per second

            if done:
                if env.score > env.highest_score:
                    env.highest_score = env.score  # Update highest score if current score is higher
                agent.update_target_model()
                print(f"Episode {episode + 1}: Score: {env.score}")
                plot_queue.put((episode + 1, env.score))

            agent.replay(batch_size)

        if (episode + 1) % save_interval == 0:
            save_count += 1
            agent.save_model('dqn_snake.pth')
            print(f"Model saved. Total saves: {save_count}")

        if episode % 10 == 0:
            print(f"Exploration rate: {agent.epsilon}")

    pygame.quit()

# Run training in a separate thread
train_thread = threading.Thread(target=train_agent)
train_thread.start()

# Start the plot update loop
while train_thread.is_alive():
    update_plot()
    time.sleep(0.1)

# Wait for the training thread to finish
train_thread.join()

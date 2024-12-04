# SnakeAI

Welcome to the **SnakeAI** repository! This project combines the classic game of Snake with advanced AI techniques to create a sophisticated and challenging gameplay experience. The repository contains two main files:

## 1. `Snake.py`
This file contains the implementation of the traditional Snake game using Pygame. The key features include:
- **Snake Class**: Handles the movement, growth, and rendering of the snake, including different body parts and directions.
- **Fruit Class**: Manages the spawning and rendering of the fruit that the snake eats to grow.
- **Main Game Logic**: Integrates the snake and fruit classes, handles collisions, game over conditions, and score display.

## 2. `snakeAI.py`
This file extends the traditional game with a Deep Q-Learning (DQL) agent to train the snake to play autonomously. Key components include:
- **Deep Q-Network (DQN)**: A neural network designed to predict the best actions the snake should take to maximize its score.
- **Environment Class**: Simulates the game environment, providing states, rewards, and handling game mechanics for the AI agent.
- **DQNAgent Class**: Manages the training of the DQL agent, including action selection, memory replay, and network updates.
- **Training Loop**: Runs multiple episodes to train the agent, saves the model periodically, and updates the gameplay in real-time.

### Prerequisites
- Python 3.x
- Pygame
- PyTorch
- NumPy
- Matplotlib

## Project Structure
- **`Snake.py`**: Traditional Snake game implementation.
- **`snakeAI.py`**: Deep Q-Learning implementation for the Snake game.
- **`Graphics/`**: Contains image assets for the game.
- **`Fonts/`**: Contains sound and font files for the game.
- **`dqn_snake.pth`**: Pre-trained model weights (if available).

## Features
- **Classic Gameplay**: Enjoy the traditional Snake game with smooth graphics and sound effects.
- **AI Training**: Watch the snake learn to play autonomously using Deep Q-Learning.
- **Real-time Plotting**: Observe the training progress with live score updates and performance graphs.

## Credit
This project was inspired by the [Google Snake Game](https://www.google.com/fbx?fbx=snake_arcade) which provided the foundation for the basic gameplay mechanics.

## Contributing
We welcome contributions! Please fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.


   

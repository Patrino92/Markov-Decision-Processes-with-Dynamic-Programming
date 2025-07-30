# Solving MDPs with Dynamic Programming 
**Assignment Year: 2024-2025**

This project solves maze-based Markov Decision Processes (MDPs) using dynamic programming techniques. The agent computes optimal policies via Value Iteration and Q-Value Iteration to navigate an environment with keys, doors, and goals.

## Overview

- **Environment**: A maze with walls, keys, doors, and goals, defined as a grid.
- **Agent Actions**: Move up, down, left, or right to achieve the maximum reward.
- **Reward System**: -1 per step, with goal rewards based on defined multipliers.
- **Algorithms**: Value Iteration and Q-Value Iteration to determine optimal state values and policies.

## Features
- Implements dynamic programming for efficient policy computation.
- Supports custom maze definitions for varied scenarios.
- Demonstrates trade-offs between exploration and exploitation.

## How to Run
1. **Setup**:
   - Place the provided files (`world.py`, `dynamic_programming.py`, `prison.txt`) in the same directory.
2. **Execution**:
   - Run `dynamic_programming.py` to execute the algorithms and test the agent.
3. **Customization**:
   - Modify `prison.txt` to create new mazes with single or multiple goals.

## Contributors
- **Kacper Nizielski**
- **Emmanouil Zagoritis**

## References
- Thomas Moerland, part of Symbolic AI course, Leiden University

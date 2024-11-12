# Project Title

This project addresses three major tasks, each focused on applying optimization and machine learning techniques to solve specific challenges. The tasks include portfolio optimization, vehicle routing, and autonomous navigation.

## Table of Contents
1. [Overview](#overview)
2. [Tasks](#tasks)
   - [Task 2: Comparative Analysis of Evolutionary Programming and Evolutionary Strategies for Portfolio Optimization](#task-2-comparative-analysis-of-evolutionary-programming-and-evolutionary-strategies-for-portfolio-optimization)
   - [Task 3: Solving the Vehicle Routing Problem with Time Windows using ACO and PSO](#task-3-solving-the-vehicle-routing-problem-with-time-windows-using-aco-and-pso)
   - [Task 4: Autonomous Taxi Navigation using Reinforcement Learning](#task-4-autonomous-taxi-navigation-using-reinforcement-learning)
3. [Installation](#installation)
4. [Usage](#usage)


## Overview

This project consists of three tasks that demonstrate the application of evolutionary strategies, metaheuristics, and reinforcement learning to solve complex optimization problems.



## Tasks

### Task 2: Comparative Analysis of Evolutionary Programming and Evolutionary Strategies for Portfolio Optimization

This task investigates evolutionary programming and evolutionary strategies for optimizing a stock portfolio. Different approaches are evaluated to achieve the best asset allocations under risk constraints.

- **Objective**: Maximize the portfolio's return-to-risk ratio.
- **Files**:
  - `(μ + λ) Evolutionary Strategies.py`
  - `(μ, λ) Evolutionary Strategies.py`
  - `advanced_ep.py`, `advanced_es.py`, `basic_ep.py`, `basic_es.py`
  - `covariance_matrix.py`, `main.py`, `monthly_returns.py`, `save_stocks_data.py`
- **Metrics**: Performance metrics include Sharpe ratio and convergence rate.

### Task 3: Solving the Vehicle Routing Problem with Time Windows using ACO and PSO

This task involves using Ant Colony Optimization (ACO) and Particle Swarm Optimization (PSO) to solve the Vehicle Routing Problem with Time Windows (VRPTW). The goal is to find optimal routes for a fleet of vehicles while adhering to specified delivery windows.

- **Objective**: Minimize total travel time and ensure adherence to delivery time windows.
- **Files**:
  - `Task_3.py`: Main script implementing ACO and PSO for VRPTW.
  - `c101.txt`: Data file containing customer locations, demands, and time window constraints.
- **Metrics**: Solution quality, computation time, and adherence to constraints.

### Task 4: Autonomous Taxi Navigation using Reinforcement Learning

This task develops an autonomous taxi navigation system using reinforcement learning in OpenAI Gym’s Taxi-v3 environment. The goal is to train an agent for efficient pick-up and drop-off operations within a grid world.

- **Objective**: Train an agent to optimize taxi navigation for customer transport.
- **Files**:
  - `Task_4.py`: Script implementing the reinforcement learning algorithm for the Taxi-v3 environment.
- **Metrics**: Reward accumulation, convergence rate, and successful completion rate.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/project_name.git
   cd project_name
   ```

2. Install the required dependencies:
 ```bash
   pip install -r requirements.txt
 ```


Usage
Task 2: Run main.py in the script_task2 folder to initiate the portfolio optimization algorithms.
Task 3: Execute Task_3.py in the script_task3 folder for VRPTW optimization using ACO and PSO.
Task 4: Run Task_4.py in the script_task4 folder to train the autonomous taxi agent.
 

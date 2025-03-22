# Starter Kit for Sony Commonsense Persona-Grounded Dialogue Challenge 2025

This repository is the Sony Commonsense Persona-Grounded Dialogue Challenge (CPDC) **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Tasks](#-tasks)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Getting Started](#-getting-started)
   - [How to write your own model?](#️-how-to-write-your-own-model)
   - [How to start participating?](#-how-to-start-participating)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
      - [How are my model responses parsed by the evaluators?](#-how-are-my-model-responses-parsed-by-the-evaluators-)
6. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Important Links](#-important-links)

# Competition Overview

You’re playing your favourite video game, navigating a bustling medieval city on your quest. When you meet a blacksmith, he greets you and mentions last night’s storm that damaged his roof. You ask about a new weapon, and he recalls your last visit, suggests an upgrade, and even offers a discount because you helped him in a previous quest.

NPCs that are context-aware respond naturally and adapt to the world around them to enable dynamic in-game interactions.

But most NPCs today have repetitive, disconnected, and robotic dialogue, struggling to balance small talk with task-driven exchanges—the very elements that make games exciting and immersive.

🎮 Enter the Commonsense Persona-grounded Dialogue Challenge (CPDC 2025)! 🎮

How can we make NPCs feel real? This challenge pushes the boundaries of AI-driven dialogue—creating characters that think, remember, and interact naturally for richer, more immersive game worlds.

This year, the challenge consists of three tasks:

- Task 1: Task-Oriented Dialogue Response Generation
- Task 2: Commonsense Dialogue Response Generation
- Task 3: A hybrid of Task 1 and Task 2, evaluating whether both objectives can be achieved simultaneously with a single model

# Dataset
We provide two dataset splits: 
- `test_evaluation_format_task*.json`: They are minimal data splits mainly for debugging. 
- `task*_train.json`: They serve as training data for the challenge. 

Each `.json` file contains several multi-turn conversations between a player and an NPC in a game environment. Each conversation has its unique worldviews, settings, player and NPC persona, etc. 

`npcdataset/` provides an interface for participants to parse the raw data. 

# Tasks
The Sony CPDC challenge will be split into three tasks. 
- Task 1: Task-Oriented Dialogue Response Generation: The data for task 1 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Participants will use this information to call functions when necessary and may use the results of these function calls to generate responses.
- Task 2: Commonsense Dialogue Response Generation: The data for task 2 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Based on this information, participants will generate natural and character-appropriate responses.
- Task 3: A hybrid of Task 1 and Task 2, evaluating whether both objectives can be achieved simultaneously with a single model. Submitting to Task 3 will automatically result in evaluation under both Task 1 and Task 2. Therefore, participants should prepare a model (or system) that meets the requirements of both tasks.



## Structure
The repo is organized as follows. 

- `agent/`: This is where you will implement your solution.  
- `npcdataset/`: This is the interface that parses the raw data. 
- `function_call_sample`: This is the interface of the function calls that will be used for Task 1 in this challenge. Note that the interface here will be different (and much simpler) than the one actually used in the evaluation, **so don't try to tamper with it**. 
    - `executor.py`: This is the interface to implement the function calls made by the agent. For the starter kit, the `Executor` will only return valid values if the agent's function calls match the `gold_functions`. Otherwise, it will not return anything. However, during the actual evaluation, `Executor` will return adequate values as long as it is a valid function call. 
- `local_run_task1.py, local_run_task2.py`: They are scripts for you to locally run your agent on the provided data. The evaluation will follow almost the same procedure.
- `task*_train.json`: They are training datasets. Each file should contain roughly 40 conversations. 
- `test_evaluation_format_task*.json`: They are subsets sampled from `task*_train.json`. Each file only contains 2 conversations so that you can quickly determine whether the agent will encouter errors. 

## Run the Starter Kit

In `agents/test_agent.py` we implement a simple baseline that directly calls LLaMA-3.1-8B-Instruct models to generate the function calls and the responses. You can run it as follows: 

```
pip install -r requirements.txt
python3 local_run_task1.py
python3 local_run_task2.py

# By default, the output will be saved in 'results/task*_responses.json'. 
```

If you want to try your own agent, follow these steps: 
- Put everything you need (including model weights---you won't have access to the Internet during evaluation) in `agents/`. 
- Implement a class `MyAgent` in `agents/my_agent.py` that has the following two methods: 
    - `generate_functions_and_responses()`, which will be called for Task 1. 
    - `generate_responses()`, which will be called for Task 2. 
- In `agents/user_config.py`, set `UserAgent = MyAgent(...)`. 
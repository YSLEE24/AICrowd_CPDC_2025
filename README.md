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

Under the challenge rules, participants can feel free to use any training data to build their solutions. 

# Tasks
The Sony CPDC challenge will be split into three tasks. 
- **Task 1: Task-Oriented Dialogue Response Generation**: The data for task 1 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Participants will use this information to call functions when necessary and may use the results of these function calls to generate responses.
- **Task 2: Commonsense Dialogue Response Generation**: The data for task 2 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Based on this information, participants will generate natural and character-appropriate responses.
- **Task 3: A hybrid of Task 1 and Task 2**, evaluating whether both objectives can be achieved simultaneously with a single model. Submitting to Task 3 will automatically result in evaluation under both Task 1 and Task 2. Therefore, participants should prepare a model (or system) that meets the requirements of both tasks.

# Evaluation Metrics 
Systems for task 1 will be evaluated on both function calling and response generation. Systems for task 2 will only be evaluated on response generation. 

To avoid overfitting the metrics, we will not disclose the exact metrics used to evaluate the systems. Also, the leaderboard will only show relative scores instead of absolute scores. 

Please refer to [local_run_task1.py](local_run_task1.py) and [local_run_task2.py](local_run_task2.py) for details on how we will run your system to get responses. 

# 🏁 Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#-how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# ✍️ How to write your own model?

Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/) to create a fork.

3.  **Clone the repository**

    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USER-NAME>/<placeholder>.git
    cd <placeholder>
    ```

4. **Install** competition specific dependencies!
    ```bash
    pip install -r requirements.txt
    ```

5. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

6. Test your model locally using `python local_evaluation.py`.

7. Accept the Challenge Rules on the main [challenge page](https://www.aicrowd.com/) by clicking on the **Participate** button. Also accept the Challenge Rules on the Task specific page (link on the challenge page) that you want to submit to.

8. Make a submission as described in [How to make a submission](#-how-to-make-a-submission) section.

## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission. 
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tracks](docs/submission.md#submitting-to-different-tracks).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.

## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission. 
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tracks](docs/submission.md#submitting-to-different-tracks).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.

## 💻 What hardware does my code run on ?
You can find more details about the hardware and system configuration in [docs/hardware-and-system-config.md](docs/hardware-and-system-config.md).
In summary, we provide you `4` x [[NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/)] in Phase 2.

Your solution will be given a certain amount of time for inference, after which it would be immediately killed and no results would be available. The time limit is set at 
| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | Track 5 |
| ------ | ------- | ------- | ------- | ------- | ------- |
| **Phase 2**| 70 minutes | 20 minutes | 30 minutes | 20 minutes | 140 minutes |

For reference, the baseline solution with zero-shot LLaMA3-8B-instruct consumes the following amount of time. 

| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | 
| ------ | ------- | ------- | ------- | ------- | 
| **Phase 2**| 1490s | 397s | 576s | 359s | 

We limit the prediction time of each sample to at most **10 seconds**. This limit applies at a batch level. For example, for a batch of 8 samples, you should return the prediction after at most 80 seconds. Otherwise, your submission will be killed. 

Your maximum repo size is 200GB. 

# ❓ Frequently Asked Questions 
## Which track is this starter kit for ?
This starter kit can be used to submit to any of the tracks. You can find more information in [docs/submission.md#submitting-to-different-tracks](docs/submission.md#submitting-to-different-tracks).

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: <placeholder>
- 🗣 Discussion Forum: <placeholder>
- 🏆 Leaderboard: <placeholder>

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
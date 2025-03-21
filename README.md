# Starter Kit for Sony Commonsense Persona-Grounded Dialogue Challenge 2025

This repo is a starter kit for the Sony Commonsense Persona-Grounded Dialogue Challenge 2025. It contains sample data and some sample code for you to locally test your agents. 

## Structure
The repo is organized as follows. 

- `agent/`: This is where you will implement your solution.  
- `npcdataset/`: This is the interface that parses the raw data. 
- `function_call_sample`: This is the interface of the function calls that will be used for Task 1 in this challenge. Note that the interface here will be different (and much simpler) than the one actually used in the evaluation, **so don't try to tamper with it**. 
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
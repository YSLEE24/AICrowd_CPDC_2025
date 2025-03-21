from typing import List, Dict, Tuple
import json
import numpy as np
import copy
import npcdataset.parsers
from agents.user_config import UserAgent
from function_call_sample import tool_map, action_map, Executor 
import argparse 
import time 
from tqdm import tqdm 
import os

def load_data(file_path):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_functions_and_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor) ->  List[Dict[str, List]]: 
    """
    Return format
    'final_function': a list of function call records which looks like this[
    {
        "name": "{function_name}",
        "parameters": {
            "{parameter}": "{item_name}"
        }
    }
    which will then be wrapped to    
    [
        {
            "turn_{turn_id}": [
                {
                    "name": "{function_name}",
                    "parameters": {
                        "{parameter}": "{item_name}"
                    }
                }
            ]
        }
    ]
    'final_response': str, which will be further wrapped to: 
    [
        {
            "turn_{turn_id}": "..."
        }
    ]
    """
    dialogue = [
        {
            "speaker": msg.speaker,
            "text": msg.text,
            "target_item": msg.target_items
        }
        for msg in cur_turn.messages
    ]
    all_results = agent.generate_functions_and_responses(
        tool_registry, 
        action_registry, 
        cur_conv.worldview,
        cur_conv.personas['npc'].to_dict(), 
        cur_conv.roles['npc'], 
        {"general_info": cur_conv.general_knowledge, "knowledge_info": cur_conv.knowledge},
        cur_conv.state, 
        dialogue, 
        executor
    )
    return all_results['final_responses']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='results/task1_responses.json')
    args = parser.parse_args()

    start_time = time.time() 

    # data_path = 'task1_train.json'
    data_path = 'test_evaluation_format_task1.json'
    data_set = load_data(data_path)
    agent = UserAgent() 

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    generated_responses = []
    for conv_idx, conversation in tqdm(enumerate(data_set)):
        cur_conv_responses = {}
        tool_registry = tool_map[conversation.function_list_id]
        action_registry = action_map[conversation.function_list_id]
        for turn_idx, turn in enumerate(conversation.turns):
            gold_functions = [
                {
                    "name": function.name,
                    "parameters": function.parameters,
                    'return': function.return_values
                }
                for function in turn.gold_functions
            ]
            cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)

            response = get_functions_and_responses(agent, conversation, turn, tool_registry, action_registry, cur_turn_exec)
            cur_conv_responses[f'turn_{turn_idx}'] = {}
            cur_conv_responses[f'turn_{turn_idx}']['response'] = response
            cur_conv_responses[f'turn_{turn_idx}']['functions'] = copy.deepcopy(cur_turn_exec.function_call_stats)
            print(f"cur_conv_functions['turn_{turn_idx}']")
            print(cur_conv_responses[f'turn_{turn_idx}'])
            print('converation', conv_idx, 'turn', turn_idx)
            print('=' * 40, '\n')
        generated_responses.append(cur_conv_responses)
    
    with open(args.save_path, 'w') as f:
        json.dump(generated_responses, f, indent=4)
    print("Responses saved to ", args.save_path)
    print("Total time spent: ", time.time() - start_time, ' seconds')

    



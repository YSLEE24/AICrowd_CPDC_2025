from typing import List, Dict, Tuple
import json
import numpy as np
import npcdataset.parsers
from agents.user_config import UserAgent
import argparse 
from tqdm import tqdm
import os
import time


def load_data(file_path):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_responses(agent, cur_conv, cur_turn, function_results) -> List[Dict[str, str]]: 
    """
        Parameters: 
            agent: UserAgent, the agent that the participants use to generate responses. 
            cur_conv: current conversation. Contains information about worldview, persona, roles, etc. 
            cur_turn: current turn, contains the conversation history up to this turn. 
            function_results: []
        Returns: str, the response of the current turn. 
    """
    dialogue = [
        {
            "speaker": msg.speaker,
            "text": msg.text,
            "target_item": msg.target_items
        }
        for msg in cur_turn.messages
    ]
    results = agent.generate_responses(
        cur_conv.worldview, 
        cur_conv.personas['npc'].to_dict(),
        cur_conv.roles['npc'],
        {"general_info": cur_conv.general_knowledge, "knowledge_info": cur_conv.knowledge},
        cur_conv.state,
        dialogue,
        function_results
    )
    return results['final_responses']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='results/task2_responses.json')
    args = parser.parse_args()

    start_time = time.time()

    # data_path = 'task2_train.json'
    data_path = 'test_evaluation_format_task2.json'
    data_set = load_data(data_path)
    agent = UserAgent()

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # task 2 does not involve function calls. 
    function_results = []
    generated_responses = []
    for conv_idx, conversation in tqdm(enumerate(data_set)):
        cur_conv_responses = {}
        for turn_idx, turn in enumerate(conversation.turns):
            response = get_responses(agent, conversation, turn, function_results)
            cur_conv_responses[f"turn_{turn_idx}"] = response 
        generated_responses.append(cur_conv_responses)

    with open(args.save_path, 'w') as f:
        json.dump(generated_responses, f, indent=4)
    print("Responses saved to ", args.save_path)
    print("Total time spent: ", time.time() - start_time)

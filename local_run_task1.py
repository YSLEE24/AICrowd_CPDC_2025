from typing import List, Dict
import json
import numpy as np
import copy
import npcdataset.parsers
from agents.user_config import UserAgent
from function_calls import tool_map, action_map, Executor 
import argparse 
import time 
from tqdm import tqdm 
import os
from evaluation_utils import extract_gold_functions, extract_predicted_functions

def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)
    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_functions_and_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor) -> List[Dict[str, List]]:
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
    data_path = 'data/task1_sample.json'
    data_set = load_data(data_path)
    agent = UserAgent() 

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    generated_responses = []
    for conv_idx, conversation in tqdm(enumerate(data_set), total=len(data_set)):
        cur_conv_responses = {"data_id": conversation.id, "outputs": []}
        tool_registry = tool_map[conversation.function_list_id]
        action_registry = action_map[conversation.function_list_id]
        for turn_idx, turn in enumerate(conversation.turns):
            gold_functions = [
                {
                    "name": function.name,
                    "parameters": function.parameters,
                    "return": function.return_values
                }
                for function in turn.gold_functions
            ]
            cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)
            response = get_functions_and_responses(agent, conversation, turn, tool_registry, action_registry, cur_turn_exec)
            cur_conv_responses["outputs"].append({
                "tool_calls": [
                    {"function": {"name": f["name"]}} for f in cur_turn_exec.function_call_stats
                ],
                "response": response
            })
        generated_responses.append(cur_conv_responses)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(generated_responses, f, indent=4, ensure_ascii=False)
    print("✅ Responses saved to:", args.save_path)
    print("⏱️ Total time spent:", round(time.time() - start_time, 2), 'seconds')

    # ✅ 정확도 평가
    with open(data_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    with open(args.save_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    gold = extract_gold_functions(sample_data)
    pred = extract_predicted_functions(result_data)

    total = 0
    correct = 0
    details = []

    for conv_id, gold_list in gold.items():
        pred_list = pred.get(conv_id, [])
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        match = gold_set & pred_set
        total += len(gold_set)
        correct += len(match)
        details.append({
            "id": conv_id,
            "gold": list(gold_set),
            "pred": list(pred_set),
            "matched": list(match)
        })

    accuracy = correct / total if total > 0 else 0

    print(f"🔍 총 Gold 함수 수: {total}")
    print(f"✅ 맞춘 함수 수: {correct}")
    print(f"🎯 정확도: {accuracy:.2%}")

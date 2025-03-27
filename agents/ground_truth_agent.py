from typing import List, Dict
import copy
import json

# This agent is used for debugging. 

class GroundTruthAgent(object):
    def __init__(self):
        ground_truth_path = 'data/task1_sample.json'
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)
        self.conversation_idx = 0
        self.turn_idx = 0

    
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor) -> Dict:
        # retrieve gold functions 
        func = self.ground_truth[self.conversation_idx][f'turn_{self.turn_idx}']['gold_functions']

        # obtain function results
        function_results = executor.execute(func)
        print(function_results)

        

        retval = {
            'prompts': 'Template', 
            'final_responses': self.ground_truth[self.conversation_idx][f'turn_{self.turn_idx}']['gold_response']
        }
        self.add_idx()
        return retval

    def add_idx(self):
        self.turn_idx += 1
        if self.turn_idx >= self.ground_truth[self.conversation_idx]['total_turn']:
            self.turn_idx = 0
            self.conversation_idx += 1
        return

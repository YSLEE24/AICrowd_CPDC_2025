from typing import List, Dict
import copy
import json

# This agent is used for debugging. 

class GroundTruthAgent(object):
    def __init__(self):
        ground_truth_path = 'test_evaluation_format_task1.json'
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)
        self.f_conversation_idx = 0
        self.f_turn_idx = 0
        self.r_conversation_idx = 0
        self.r_turn_idx = 0
    

    def generate_functions(self, tool_functions, action_functions, dialogue) -> Dict: 
        """
        Parameters
        ----------
        tool_functions: str
        action_functions: str
        dialogue: List[Dict[str, str]]

        Return
        ----------
        {
            "prompts": "...",
            "final_responses": [
                {
                    "name": "...",
                    "parameters": [
                        "{name}": "{value}"
                    }
                }
            ]
        }
        """
        print(self.f_conversation_idx, self.f_turn_idx)
        func = self.ground_truth[self.f_conversation_idx][f'turn_{self.f_turn_idx}']['gold_functions']
        self.add_fidx()
        return {'prompt':'Placeholder', 'final_functions': func}
    
    def generate_responses(self, worldview, persona, role, knowledge, state, dialogue, function_results) -> Dict[str, str]:
        """
        Parameters
        ----------
        worldview: str
        persona: Dict[str, str]
        role: str
        knowledge: Dict[str, Any]
        state: Dict[str, str]
        dialogue: List[Dict[str, str]]
        function_results: List[Dict[str, Any]]

        Return
        ----------
        Dict[str, str]
            {
                "prompts": "..."
                "final_responses": "..."
            }
        """
        print(self.r_conversation_idx, self.r_turn_idx)

        resp = self.ground_truth[self.r_conversation_idx][f'turn_{self.r_turn_idx}']['gold_response']
        self.add_ridx()
        return {'prompts': resp, 'final_responses': resp}

    def add_ridx(self):
        self.r_turn_idx += 1
        if self.r_turn_idx >= self.ground_truth[self.r_conversation_idx]['total_turn']:
            self.r_turn_idx = 0
            self.r_conversation_idx += 1
        return

    def add_fidx(self):
        self.f_turn_idx += 1
        if self.f_turn_idx >= self.ground_truth[self.f_conversation_idx]['total_turn']:
            self.f_turn_idx = 0
            self.f_conversation_idx += 1
        return
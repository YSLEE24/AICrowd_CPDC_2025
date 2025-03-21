from typing import Any, List, Dict


class DummyResponseAgent(object):
    def __init__(self):
        """ Load your model(s) here """
        pass
    
    
    
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        """
        Parameters
        ----------
        tool_registry, action_registry are function registries that can index functions with names
        worldview: str
        persona: Dict[str, str]
        role: str
        knowledge: Dict[str, Any]
        state: Dict[str, str]
        dialogue: List[Dict[str, str]]
        executor: a module that can execute function calls and record history of function calling. 

        Return
        ----------
        Dict[str, str]
            {
                "prompts": "..."
                "final_responses": "..."
            }
        """
        function = {
            "prompts": "",
            "final_functions": []
        }
        return function
    
    
    
    def generate_responses(self, worldview: str, persona: Dict[str, str], role: str, knowledge: Dict[str, Any],
                           state: Dict[str, str], dialogue: List[Dict[str, str]], function_results: List[Dict[str, Any]]) -> Dict[str, str]:
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
        response = {
            "prompts": "",
            "final_responses": "THIS IS A TEST REPLY"
        }
        return response

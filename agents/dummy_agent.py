from typing import Any, List, Dict


class DummyResponseAgent(object):
    def __init__(self):
        """ Load your model(s) here """
        pass
    
    
    
    def generate_functions(self, tool_functions: str, action_functions: str, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
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

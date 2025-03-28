from typing import List, Dict
import copy
import os
from openai import OpenAI
import json
# comment out that line before pushing. 


class OpenAIAgent(object):
    def __init__(self):
        """Initialize an openai agent"""
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


    def _create_messages_for_function(self, tool_functions, action_functions, dialogue):
        """
            We need to create tools as follows: 
             [{
                "type": "function",
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                }
            }]
            tools: a list of dicts. 
                tools[0]: a dict
                    'type': 'function', 
                    'name': function_name, 
                    'description': function_description
                    'strict': encouraged to be true. 
                    'parameters': a dict: 
                        'type': 'object', 
                        'properties': a dict
                            parameter one name: a dict
                                'type': type of this parameter
                                'description': description of this parameter
                            parameter two name: ...
                        'required': a list, containing the names of required parameters. 
                        'additionalProperties': False

        """
        all_tools = []
        
        for k in tool_functions['function_registry']: 
            cur_tool = {
                'type': 'function', 
                'name': k, 
                'strict': True, 
            }
            

            param_dict = {
                'type': 'object', 
                'properties': {},
                'additionalProperties': False
            }
            docstring_lines = tool_functions['function_registry'][k]['description'].split('\n')
            desc_lines = []
            line_idx = 0
            while 1:
                if docstring_lines[line_idx] != '':
                    desc_lines.append(docstring_lines[line_idx])
                    line_idx += 1
                else:
                    break 
            cur_tool['description'] = ' '.join(desc_lines)

            while 1:
                if 'Parameters' in docstring_lines[line_idx]: 
                    line_idx += 2
                    break
                else:
                    line_idx += 1 
            # begin of parameters 
            cur_param = {}
            required_list = []
            while 1: 
                if 'Returns' in docstring_lines[line_idx]:
                    break
                param_name, param_type = docstring_lines[line_idx].split(':')
                if 'List' in param_type: 
                    param_type = 'array'
                elif 'str' in param_type: 
                    param_type = 'string'
                cur_param = {
                    'type': param_type
                }
                
                line_idx += 1
                description_list = []
                while 1:
                    if docstring_lines[line_idx] == '':
                        line_idx += 1
                        break 
                    else:
                        description_list.append(docstring_lines[line_idx])
                        line_idx += 1
                description_text = ' '.join(description_list)
                cur_param['description'] = description_text
                param_dict['properties'][param_name] = cur_param
                required_list.append(param_name)
            param_dict['required'] = required_list
            cur_tool['parameters'] = param_dict


            all_tools.append(cur_tool)

        for k in action_functions['function_registry']: 
            cur_action = {
                'type': 'function', 
                'name': k, 
                'strict': True, 
            }
            

            param_dict = {
                'type': 'object', 
                'properties': {},
                'additionalProperties': False
            }
            docstring_lines = action_functions['function_registry'][k]['description'].split('\n')
            desc_lines = []
            line_idx = 0
            while 1:
                if docstring_lines[line_idx] != '':
                    desc_lines.append(docstring_lines[line_idx])
                    line_idx += 1
                else:
                    break 
            cur_action['description'] = ' '.join(desc_lines)
            # print(cur_action['description'])

            while 1:
                if 'Parameters' in docstring_lines[line_idx]: 
                    line_idx += 2
                    break
                else:
                    line_idx += 1 
            # begin of parameters 
            cur_param = {}
            required_list = []
            while 1: 
                if 'Returns' in docstring_lines[line_idx]:
                    break
                param_name, param_type = docstring_lines[line_idx].split(':')
                if 'List' in param_type: 
                    cur_param = {
                        'type': 'array', 
                        'items': {
                            'type': 'string'
                        }
                    }
                elif 'str' in param_type: 
                    param_type = 'string'
                    cur_param = {
                        'type': param_type
                    }


                
                line_idx += 1
                description_list = []
                while 1:
                    if docstring_lines[line_idx] == '':
                        line_idx += 1
                        break 
                    else:
                        description_list.append(docstring_lines[line_idx])
                        line_idx += 1
                description_text = ' '.join(description_list)
                cur_param['description'] = description_text
                param_dict['properties'][param_name] = cur_param
                required_list.append(param_name)
            param_dict['required'] = required_list
            cur_action['parameters'] = param_dict


            all_tools.append(cur_action)

        system_prompt = """# Instruction
You are an assistant in estimating function names and arguments.
You need information to answer the text entered by the user.
Use the following steps to perform function's estimation.
1. From the given function information, select a function that can obtain the information you need.
2. Make the arguments needed by the function as appropriate.

## Additional Information
{}"""
        additional_info = ""
        for info in dialogue[-1]["target_item"]:
            if additional_info == "":
                additional_info = "parameter name: name, value: " + info["name"]
            else:
                additional_info = additional_info + "\nargument name: name, value: " + info["name"]
        input_messages = [{"role": "system", "content": system_prompt.format(additional_info)}, 
            {'role': 'user', 'content': dialogue[-1]['text']}]


        return all_tools, input_messages

    def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results): 
        base_prompt = """\
You are an assistant who becomes a character.
Use the following character settings and knowledge to create your response.

# Character Setting
[Here is Character Setting]

# Knowledge
[Here is Knowledge]"""

        worldview = worldview + "\n" + knowledge["general_info"]
        
        character_setting = ""
        for key in persona:
            if character_setting == "":
                character_setting = key + ": " + persona[key]
            else:
                character_setting = character_setting + "\n" + key + ": " + persona[key]

        knowledge_setting = ""
        for f_result in function_results:
            if knowledge_setting != "":
                knowledge_setting = knowledge_setting + "\n"
            for arg in f_result["parameters"]:
                if knowledge_setting == "":
                    knowledge_setting = arg + ": " + str(f_result["parameters"][arg])
                else:
                    knowledge_setting = knowledge_setting + ", " + arg + ": " + str(f_result["parameters"][arg])
            for item in f_result["return"]:
                if f_result['name'] == 'sell':
                    print(f_result['return'])
                for key in item:
                    if knowledge_setting == "":
                        knowledge_setting = key + ": " + item[key]
                    else:
                        knowledge_setting = knowledge_setting + ", " + key + ": " + item[key]
                        
        for item in knowledge["knowledge_info"]:
            if knowledge_setting != "":
                knowledge_setting = knowledge_setting + "\n"
            for key in item:
                if knowledge_setting == "":
                    knowledge_setting = key + ": " + item[key]
                else:
                    knowledge_setting = knowledge_setting + ", " + key + ": " + item[key]

                    
        history_list = []
        for item in dialogue:
            role = "user"
            if item["speaker"] == "npc":
                role = "assistant"
            history_list.append({"role":role, "content":item["text"]})

        prompt = base_prompt.replace("[Here is Worldview]", worldview)
        prompt = prompt.replace("[Here is Role]", role)
        prompt = prompt.replace("[Here is Character Setting]", character_setting)
        prompt = prompt.replace("[Here is Knowledge]", knowledge_setting)
        
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages

    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor) -> Dict: 
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


                
        tools, messages = self._create_messages_for_function(tool_registry, action_registry, dialogue)
        # call openai api to generate functions 
        response = self.client.responses.create(
            model="gpt-4o",
            input=messages,
            tools=tools,
        )
        all_functions = []
        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.arguments)
            all_functions.append({
                'name': name, 
                'parameters': args
            })
        # obtain results 
        function_results = executor.execute(all_functions)
        print(function_results)


        # then generate response
        messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        response = self.client.responses.create(
            model="gpt-4o",
            input=messages_resp,
        )
        return {
            'prompts': 'Placeholder', 
            'final_responses': response.output_text
        }
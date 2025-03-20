from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy

class TestResponseAgent(object):
    def __init__(self):
        model_path = 'meta-llama/Llama-3.1-8B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.max_seq_len = 2048
        self.max_gen_len = 1000
    
    def _create_messages_for_function(self, tool_functions, action_functions, dialogue):
        """
            Formats function information into prompt. 
            'tool_functions' and 'action_functions' are both function registries
            that supports indexing a function with its name
        """
        base_prompt = """# Instruction
You are an assistant in estimating function names and arguments.
You need information to answer the text entered by the user.
Use the following steps to perform function's estimation.
1. From the function information below, select a function that can obtain the information you need.
2. Make the arguments needed by the function as appropriate.
The format of the output is:
function name: xxx
argument name: xxx, vlaue: xxx

## Function Information
{}

## Additional Information
{}"""
        function_information = []
        for tool_name in tool_functions.all_functions():
            tool_ = tool_functions.get_function(tool_name)
            tool_prompt = """
# Function Name: {}
# Function Docstring: 
{}
            """.format(tool_['name'], tool_['description'])
            function_information.append(tool_prompt)
        for action_name in action_functions.all_functions():
            action_ = action_functions.get_function(action_name)
            action_prompt = """
# Function Name: {}
# Function Docstring: 
{}
            """.format(action_['name'], action_['description'])
            function_information.append(action_prompt)
        function_information_agg = '\n'.join(function_information)
        additional_info = ""
        for info in dialogue[-1]["target_item"]:
            if additional_info == "":
                additional_info = "parameter name: name, value: " + info["name"]
            else:
                additional_info = additional_info + "\nargument name: name, value: " + info["name"]
        # TODO: Why sometimes 'argument name' and sometimes 'parameter name'? 
        input_text = dialogue[-1]["text"]
        prompt = base_prompt.format(function_information_agg, additional_info)
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.append({"role":"user", "content":input_text})
        # print("PROMPT: ")
        # print(prompt)
        # print('=' * 40)
        return messages

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
                    knowledge_setting = arg + ": " + f_result["parameters"][arg]
                else:
                    knowledge_setting = knowledge_setting + ", " + arg + ": " + f_result["parameters"][arg]
            for item in f_result["return"]:
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
        
        # first ask the LLM to generate the functions to call. 
        messages_func = self._create_messages_for_function(tool_registry, action_registry, dialogue)
        input_ids = self.tokenizer.apply_chat_template(
            messages_func, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None, 
            max_new_tokens=self.max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        res = outputs[0][input_ids.shape[-1]:]        
        items = self.tokenizer.decode(res, skip_special_tokens=True).split("\n")

        # parse the generated results. 
        final_functions = []
        res_item = {}
        for item in items:
            if "function name: " in item:
                if "name" in res_item:
                    final_functions.append(copy.deepcopy(res_item))
                    res_item = {}
                res_item["name"] = item.replace("function name: ", "").split(",")[0]
                res_item["parameters"] = {}
            elif "argument name: " in item:
                arg_name = ""
                arg_val = ""
                if "value: " in item:
                    arg_name = item.split("value: ")[0].replace("argument name: ", "").split(",")[0]
                    arg_val = item.split("value: ")[1]

                if arg_name != "":
                    if not "parameters" in res_item:
                        res_item["parameters"] = {}
                    res_item["parameters"][arg_name] = arg_val
                    
        if "name" in res_item:
            final_functions.append(res_item)

        # obtain function_calling results via the executor
        function_results = executor.execute(final_functions)



        # then generate response. 
        messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        input_ids = self.tokenizer.apply_chat_template(
            messages_resp,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=self.max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        res = outputs[0][input_ids.shape[-1]:]
        res_str = self.tokenizer.decode(res, skip_special_tokens=True).replace("\n", " ")

        # However, participants can do more than that, like back and forth calling of functions. 

        # Only return response. We don't return function calls as it is recorded in the 'executor'. 
        return {
            'prompts': messages_resp, 
            'final_responses': res_str
        }
        
        
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
        messages = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=self.max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        res = outputs[0][input_ids.shape[-1]:]
        res_str = self.tokenizer.decode(res, skip_special_tokens=True).replace("\n", " ")
        
        response = {
            "prompts": messages,
            "final_responses": res_str
        }
        return response

from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy

class VanillaLlamaAgent(object):
    """
        VanillaLlamaAgent is a simple agent implementation for the GPU track of the Sony CPDC 2025 Challenge. 

        This agent takes in information from the dialogue, the functions, and the background of the scenario, 
        It calls a LLaMA-3.1-8B-Instruct model twice per turn. 
        The first call aims to find appropriate functions to call, 
        and the second generates responses. 

        Attributes: 
            model: The LLaMA-3.1-8B-Instruct model. 
            tokenizer: The tokenizer for the LLaMA-3.1-8B-Instruct model. 
        
        Remember to specify all HF models you need in `aicrowd.json`. 
        Otherwise, the evaluator will not be able to download the models, and your submission will fail. 
    """
    def __init__(self):
        """Load necesasry models and configurations here"""
        # Please specify all HF model paths in `aicrowd.json`. 
        model_path = 'meta-llama/Llama-3.1-8B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.max_seq_len = 2048
        self.max_gen_len = 50

    ############################################################
    # The entrypoint of the evaluator.  
    ############################################################
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        """
        Generates the responses given the dialogue, the functions, and the background of the video game scenario.

        This method is the entry point called by the evaluator. It is implemented with the following steps: 
        
        1. Prepare prompts for function calling. 
        2. Call the model (LLaMA-3.1-8B-Instruct) to generate the necessary function calls. 
        3. Use the `executor` to obtain the function call results. 
        4. With the function call results and the background, prepare prompts for response generation. 
        5. Call the model (LLaMA-3.1-8B-Instruct) to generate the text response. 

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action functions (OpenAI function calling format). 
            Implementations can be found in the directory `function_calls`. 
            
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario. 
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn. 
            executor: This is implemented in `function_calls/executor.py`. It takes in a list of function calls and return the results. 

        Returns: 
            A dict with the following keys: 
                'final_responses': str, the text responses. 

            You do not have to return the function calls. The `executor` will record all the function calls that are passed to it. 
        """

        
        # Step 1: In our first call to the LLM, we ask it to generate all necessary functions to call.  
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

        # Step 2: Parse the generated function call results.  
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

        # Step 3: Obtain the return values for all function call results via the executor. 
        function_results = executor.execute(final_functions)

        # Step 4: Based on the function call results, generate response. 
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

    ############################################################
    # Helper functions. 
    ############################################################

    def _create_messages_for_function(self, tool_functions, action_functions, dialogue):
        """
            Creates the messages to feed to the model to generate the necessary function calls. 

            Args: 
                tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
                action_registry: A dict mapping action names to action functions (OpenAI function calling format). 
                Implementations can be found in the directory `function_calls`. 
                dialogue: List[Dict], the dialogue history. dialogue[-1] refers to the current turn. 
    
            Returns: 
                messages: List[Dict], the messages to feed to the model. 
            
            Note that the prompt here is not tuned for performance. 
        """
        function_prompt = (
            "# Instruction\n"
            "You are an assistant in estimating function names and arguments given some dialogues in a video game world.\n"
            "You will need the following information to respond to the user's input. \n"
            "Use the following steps to estimate the necessary function names and arguments. \n"
            "\n"
            "1. Read the dialogue and the target item. \n"
            "2. From the given function information, select the functions that can obtain the information you need. \n"
            "3. Fill in the arguments needed by the function as appropriate. \n"
            "The format of the output is:\n"
            "function name: xxx\n"
            "argument name: xxx, value: xxx\n"
            "\n"
            "Note: You may select multiple functions or no functions at all. \n"
            "\n"
            "# Function Information\n"
            "{}\n"
            "# Additional Information\n"
            "{}\n"
        )

        # Prepare function information by concatenating all function names and docstrings. 
        function_information = []
        for tool_name in tool_functions['function_registry'].keys():
            tool_ = tool_functions['function_registry'][tool_name]
            tool_prompt = (
                "# Function Name: {}\n"
                "# Function Docstring: {}\n"
            ).format(tool_['name'], tool_['description'])
            function_information.append(tool_prompt)
        for action_name in action_functions['function_registry'].keys():
            action_ = action_functions['function_registry'][action_name]
            action_prompt = (
                "# Function Name: {}\n"
                "# Function Docstring: {}\n"
            ).format(action_['name'], action_['description'])
            function_information.append(action_prompt)
        function_information_agg = '\n'.join(function_information)

        # 'target_item' is used to indicate what the user is referring to, such as 'this', 'that', 'the one', etc. 
        additional_info = ""
        if len(dialogue[-1]["target_item"]) > 0:
            additional_info = "In the dialogue, the user may be referring to the following items: \n"
            for info in dialogue[-1]["target_item"]:
                additional_info += f'parameter name: name, value: {info["name"]}\n'

        input_text = dialogue[-1]["text"]
        prompt = function_prompt.format(function_information_agg, additional_info)
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.append({"role":"user", "content":input_text})

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
                    knowledge_setting = arg + ": " + str(f_result["parameters"][arg])
                else:
                    knowledge_setting = knowledge_setting + ", " + arg + ": " + str(f_result["parameters"][arg])
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
    

        
        

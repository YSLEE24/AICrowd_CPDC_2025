from typing import List, Dict
import copy
import os
from openai import OpenAI
import json



class NewOpenAIAgent(object):
    """
    NewOpenAIAgent is a new implementation for API Track of the Sony CPDC 2025 Challenge. 
    
    This agent takes in information from the dialogue, the functions, and the background of the scenario, 
    e.g. worldview, persona, role, etc., and generates corresponding function calls and text responses. 
    The function information has already been converted from langchain.tools to the OpenAI function calling format 
    using the method `convert_to_openai_function`, to simplify the implementation. 
    For details of the function calls, please refer to the `function_calls` directory. 

    Attributes: 
        client: OpenAI client. 
                You can assume that the API keys are automatically configured in the environment variables. 
    """
    def __init__(self):
        """
        Initialize an openai agent. You can assume that the API keys are automatically configured in the environment variables. 
        """
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.environ.get("TEMPERATURE", 0.7))
        self.top_p = float(os.environ.get("TOP_P", 0.9))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", 200))
        self.MAX_TOKENS_FUNCTION_CALL=2000


    ############################################################
    # The entrypoint of the evaluator.  
    ############################################################
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, npc_persona, role, knowledge, state, dialogue, executor): 
        try:
            function_results = [] # only for task2
            messages_resp = self._create_messages_for_dialogue(worldview, npc_persona, role, knowledge, state, dialogue, function_results)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_resp,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = "Sorry, I encountered an error generating a response."
            print(f"[ERROR] LLM call failed: {e}")

        return {"final_responses": reply}


    ############################################################
    # Helper functions. 
    ############################################################

    def _prepare_openai_functions(self, tool_registry, action_registry):
        openai_tool_functions = list(tool_registry['function_registry'].values())
        openai_action_functions = list(action_registry['function_registry'].values())
        return openai_tool_functions + openai_action_functions



    def _create_messages_for_function(self, tool_registry, action_registry, worldview, npc_persona, role, knowledge, state, dialogue):
        """
        Creates the messages to feed to OpenAI client to generate the necessary function calls. 

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action functions (OpenAI function calling format). 
            Implementations can be found in the directory `function_calls`. 
            dialogue: List[Dict], the dialogue history. dialogue[-1] refers to the current turn. 

        Returns: 
            input_messages: List[Dict], the messages to feed to OpenAI client. 
            all_functions: List[Dict], a list of functions in the OpenAI function calling format. 
        """
        # 1) Build & validate function definitions
        all_functions = self._prepare_openai_functions(tool_registry, action_registry)

        # 최신 형식 강제 적용
        validated = []
        seen = set()
        for fn in all_functions:
            base = fn.get("function", fn)
            if base["name"] in seen:
                continue
            seen.add(base["name"])
            validated.append({
                "type": "function",
                "function": {
                    "name": base["name"],
                    "description": base.get("description", ""),
                    "parameters": base["parameters"]
                }
            })
        all_functions = validated

                
        # 3. 프롬프트 보수적 재작성 + Mentioned Items 포함
        last = dialogue[-1] if dialogue else {}
        items = last.get("target_item") or []
        mentioned_items_text = ""
        if isinstance(items, list) and items:
            names = [i["name"] for i in items if isinstance(i, dict) and "name" in i]
            if names:
                mentioned_items_text = (
                    "\n## Mentioned Item Names from this turn:\n" + "\n".join(f"- {n}" for n in names) + "\n"
                )

        # npc_identity = f"You are NPC {npc_persona['name']}, a {npc_persona['occupation']} working as {role}."
        # world_context = f"Setting: {state.get('place', '')}, Date: {state.get('date', '')}"
        system_prompt = f"""
You are an assistant responsible for determining which functions to call based on player-NPC dialogue in a video game.

🧠 Function Calling Rules:
You will need the following information to respond to the user's input. 
Use the following steps to estimate the necessary function names and arguments.

1. Read the dialogue and the target item.  
2. From the given function information, select the functions that can obtain the information you need.  
3. Fill in the arguments needed by the function as appropriate.  

- Prefer check_* functions (like check_quest_status, check_reward) if a player refers to a known quest or item.
- If multiple functions are possible, prefer **'check_' type functions** over others (e.g., 'get_reward'). These are more conservative and factual.
- Choose the **most relevant and minimal** function.

🎯 Parameter Selection Rules:
- **ONLY include parameters that are explicitly mentioned or strongly implied in the user's request**
- For `search_` functions: **Only provide clearly mentioned or strongly implied necessary parameters**  
- Do NOT guess or include empty/default parameters.
- **Be precise and minimal**: incorrect or excessive parameters will cause function calls to fail.  

- Examples:
  - If the player says "Tell me about 'Dragon Hunt' quest", include: `quest_name=\"Dragon Hunt\"`
  - If the player says "Am I strong enough at level 10?", include: `player_level=10`

{mentioned_items_text}

If you're unsure, it is better to attempt a reasonable function call with minimal parameters.
"""


        messages = [{"role": "system", "content": system_prompt}]
        for turn in dialogue:
            role = "user" if turn["speaker"] == "player" else "assistant"
            messages.append({"role": role, "content": turn.get("text", "")})


        return messages, all_functions



    def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results):
        """
        Based on the background information of the video game and the dialogue history, 
        creates the messages to feed to OpenAI client to generate the text response. 

        Args: 
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario. 
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn. 
            function_results: A list of function call results. 
        """
        dialogue_prompt = (
            "# Instruction\n"
            "You are not an assistant. You are a character in a fantasy video game.\n"
            "Stay consistent with your character’s personality and quirks, even when unsure.\n"
            "You may show light emotion or subtle behavior if it fits the moment.\n"
            "Speak naturally and briefly, as if talking to another adventurer—not like an AI.\n"
            "\n"
            "# Guidelines for your response\n"
            "- Respond in 1 to 2 short, direct sentences.\n"
            "- When it feels natural, echo the tone or sentence structure of the player.\n"
            "- Prioritize relevant in-world knowledge, not poetic or dramatic commentary.\n"
            "- Include subtle actions or concrete examples if they help make the point.\n"
            "- Avoid filler or long explanations unless the player’s intent is unclear.\n"
            "- Never break character or sound like an assistant.\n"
            "\n"
            "# Character Settings:\n"
            "{}\n"
            "\n"
            "# General Knowledge of All Items\n"
            "{}\n"
            "\n"
            "# Worldview\n"
            "{}\n"
            "\n"
            "# State\n"
            "{}\n"
        )
        worldview = worldview + '\n' + knowledge['general_info']

        # 'persona' is a dict that specifies properties of the character. 
        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'
        
        # function_knowledge records the specific knowledge obtained from the function calls. 
        function_knowledge = ""
        for f_result in function_results:
            # record each function call in the following format: 
            # function_name: parameter_name1, parameter_name2, ... -> return_value1, return_value2, ...
            parameter_info = []
            return_value_info = []
            for arg in f_result["parameters"]:
                parameter_info.append(f'{arg}: {str(f_result["parameters"][arg])}')
            parameter_info = ", ".join(parameter_info)

            for item in f_result["return"]:
                return_value_info.append(str(item))
            return_value_info = ", ".join(return_value_info)
            function_knowledge += f'{f_result["name"]}: {parameter_info} -> {return_value_info}\n'

        
        # general_knowledge records the general knowledge of all items involved in the dialogue. 
        general_knowledge = ""
        for item in knowledge["knowledge_info"]:
            item_info = []
            for key in item:
                item_info.append(f'{key}: {item[key]}')
            item_info = ", ".join(item_info)
            general_knowledge += f'{item_info}\n'
        
        # prepare the dialogue history. 
        history_list = []
        for item in dialogue:
            # The dataset uses 'npc' to indicate the characters in the game. 
            role = "user"
            if item["speaker"] == "npc":
                role = "assistant"
            history_list.append({"role":role, "content":item["text"]})
        

        state_info = ""
        for k, v in state.items():
            state_info += f'- {k}: {v}\n'
        

        # prompt = dialogue_prompt.format(character_setting, function_knowledge, general_knowledge, worldview)
        prompt = dialogue_prompt.format(character_setting, function_knowledge, general_knowledge, worldview, state_info)
        
        # print('===============dialog prompt=============')
        # print(print)

        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages



# import os
# import json
# import pickle
# import numpy as np
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer


# class NewOpenAIAgent(object):
#     def __init__(self):
#         self.client = OpenAI(
#             api_key=os.environ.get("OPENAI_API_KEY"),
#             base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
#         )

#         # ✅ 유사도 기반 함수 필터링
#         with open("data/tool_embeddings.pkl", "rb") as f:
#             self.tool_embeddings = pickle.load(f)

#         # model = SentenceTransformer("all-MiniLM-L6-v2")
#         model_name="all-MiniLM-L6-v2"
#         local_dir="data/models"
#         model_path = os.path.join(local_dir, model_name)
        
#         # 경로가 없으면 HuggingFace에서 다운로드 후 저장
#         if not os.path.exists(model_path):
#             print(f"📥 Downloading {model_name} to {model_path}...")
#             self.model = SentenceTransformer(model_name)
#             self.model.save(model_path)
#         else:
#             print(f"📦 Loading {model_name} from {model_path}...")
#             self.model = SentenceTransformer(model_path)

#         self.max_token = 70

#     def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):

#         function_results = []

#         # Step 4: 텍스트 생성 메시지 준비
#         messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        
#         # Step 5: 텍스트 생성
#         response = self.client.responses.create(
#             model="gpt-4o-mini",
#             input=messages_resp,
#             max_tokens = self.max_token
#         )
#         return {
#             'final_responses': response.output_text
#         }
    

#     def _prepare_openai_functions(self, tool_registry, action_registry):
#         openai_tool_functions = []
#         for tool_name, tool_function in tool_registry["function_registry"].items():
#             tool_function["type"] = "function"
#             tool_function["name"] = tool_name
#             openai_tool_functions.append(tool_function)

#         openai_action_functions = []
#         for action_name, action_function in action_registry["function_registry"].items():
#             action_function["type"] = "function"
#             action_function["name"] = action_name
#             openai_action_functions.append(action_function)

#         print("\n===== 🔧 FUNCTIONS PASSED TO GPT (tools=) =====")
#         for f in openai_tool_functions + openai_action_functions:
#             print(f"🔹 name: {f.get('name')}, description: {f.get('description')}")
#         return openai_tool_functions + openai_action_functions

#     def _create_messages_for_function(self, tool_registry, action_registry, dialogue):
#         all_functions = self._prepare_openai_functions(tool_registry, action_registry)

#         function_prompt = (
#             "# Instruction\n"
#             "You are an assistant in estimating function names and arguments given some dialogues in a video game world.\n"
#             "You will need the following information to respond to the user's input.\n"
#             "Use the following steps to estimate the necessary function names and arguments.\n"
#             "1. Read the dialogue and the target item.\n"
#             "2. From the given function information, select the functions that can obtain the information you need.\n"
#             "3. Fill in the arguments needed by the function as appropriate.\n"
#             "Note: You may select multiple functions or no functions at all.\n"
#             "# Additional Information\n{}\n# Dialogue\n"
#             "The user input for the current turn is as follows.\n"
#         )

#         additional_info = ""
#         if len(dialogue[-1]["target_item"]) > 0:
#             additional_info = "In the dialogue, the user may be referring to the following items:\n"
#             for info in dialogue[-1]["target_item"]:
#                 additional_info += f'parameter name: name, value: {info["name"]}\n'

#         input_messages = [{"role": "system", "content": function_prompt.format(additional_info)},
#                           {'role': 'user', 'content': dialogue[-1]['text']}]

#         return input_messages, all_functions

#     def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results):

#         dialogue_prompt = (
#             "# Instruction\n"
#             "You are an assistant that plays the role of a character in a video game.\n"
#             "Follow the character's personality, background, and role strictly.\n"
#             "You must respond naturally and consistently, as if you are the character.\n\n"
#             "- Your response must be under 70 tokens.\n"
#             "# Character Settings:\n{}\n\n"
#             "# Role Info\n{}\n"
#             "# General Knowledge of All Items\n{}\n"
#             "# Worldview\n{}\n"
#             "# State Info\n{}\n"
#         )

#         character_setting = '\n'.join(f'- {k}: {v}' for k, v in persona.items())

#         if isinstance(role, dict):
#             role_info = '\n'.join(f'- {k}: {v}' for k, v in role.items())
#         else:
#             role_info = str(role)


#         general_knowledge = '\n'.join(
#             ', '.join(f'{k}: {v}' for k, v in item.items())
#             for item in knowledge["knowledge_info"]
#         )

#         worldview_text = worldview + '\n' + knowledge.get('general_info', '')

#         state_info = '\n'.join(f'- {k}: {v}' for k, v in state.items())

#         history_list = [
#             {"role": "assistant" if item["speaker"] == "npc" else "user", "content": item["text"]}
#             for item in dialogue
#         ]

#         prompt = dialogue_prompt.format(
#             character_setting,
#             role_info,
#             general_knowledge,
#             worldview_text,
#             state_info
#         )

#         messages = [{"role": "system", "content": prompt}] + history_list
#         return messages

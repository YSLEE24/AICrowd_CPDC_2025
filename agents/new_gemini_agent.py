import google.generativeai as genai
# from google.generativeai.types import Content
import json
import os
import time

class NewGeminiAgent(object):
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
        # self.client = OpenAI(
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        #     base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # )

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY")) 
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # 호출 통계용 변수 초기화
        self.call_count = 0
        self.call_times = []  # 각 호출 시각 저장

    ############################################################
    # The entrypoint of the evaluator.  
    ############################################################
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        # 호출 통계 업데이트
        self.call_count += 1
        now = time.time()
        self.call_times.append(now)
        
        # 1분 이내 호출만 필터링
        one_min_ago = now - 60
        recent_calls = [t for t in self.call_times if t > one_min_ago]
        calls_per_min = len(recent_calls)

        # 디버깅 출력
        print(f"[Gemini 호출] 총 호출 횟수: {self.call_count}, 최근 1분 호출 횟수: {calls_per_min}")

        # Step 1: Function Calling용 메시지 생성
        function_messages, all_functions = self._create_messages_for_function(tool_registry, action_registry, dialogue)
        function_prompt = "\n".join([m["content"] for m in function_messages])

        print("=====================")
        print(function_prompt)
        print("=====================")

        # Step 2: Gemini로 함수 추론
        response = self.model.generate_content(function_prompt)
        time.sleep(4)  # 요청 간 딜레이 추가

        # Step 3: 함수 파싱 및 실행
        functions_to_call = self._extract_function_calls(response.text)
        function_results = executor.execute(functions_to_call)

        # Step 4: 응답 생성용 메시지 구성
        messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        full_prompt = "\n".join([m["content"] for m in messages_resp])

        print("=====================")
        print(full_prompt)
        print("=====================")

        # Step 5: Gemini로 응답 생성
        response = self.model.generate_content(full_prompt)
        time.sleep(4)  # 요청 간 딜레이 추가

        # 호출 통계 업데이트 (응답 생성도 한 번의 호출이므로)
        self.call_count += 1
        now2 = time.time()
        self.call_times.append(now2)

        # 다시 1분당 호출 확인
        recent_calls2 = [t for t in self.call_times if t > now2 - 60]
        calls_per_min2 = len(recent_calls2)

        print(f"[Gemini 호출] 총 호출 횟수: {self.call_count}, 최근 1분 호출 횟수: {calls_per_min2}")

        return {
            'final_responses': response.text
        }
    
    def _extract_function_calls(self, response_text):
        """
        Gemini는 function calling을 지원하지 않으므로,
        응답 텍스트에서 function 이름과 파라미터를 파싱해야 함
        → 현재는 임시 파서. 필요 시 개선 가능
        """
        # 예시 텍스트: call check_basic_info(name='Avis Wind')
        import re
        functions = []
        matches = re.findall(r"call\s+(\w+)\((.*?)\)", response_text)
        for name, args_str in matches:
            try:
                args = json.loads("{" + args_str + "}")
            except:
                args = {}
            functions.append({'name': name, 'parameters': args})
        return functions

    ############################################################
    # Helper functions. 
    ############################################################

    def _prepare_openai_functions(self, tool_registry, action_registry):
        """
        Prepare the list of functions as inputs to the OpenAI API. 
        The values of `tool_registry` and `action_registry` have already been converted to the OpenAI function calling format. 

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action function (OpenAI function calling format). 
            Implementations can be found in the directory `function_calls`. 

        Returns: 
            openai_functions: List[Dict], a list of functions in the OpenAI function calling format. 
        """
        openai_tool_functions = []
        for tool_name, tool_function in tool_registry['function_registry'].items():
            tool_function['type'] = 'function'
            # With my openai=1.77.0 and langchain=0.3.25 version, this should be manually added. 
            # Please note that this may not be necessary for all OpenAI and langchain versions. 
            # Please test it before submission. 
            openai_tool_functions.append(tool_function)

        openai_action_functions = []
        for action_name, action_function in action_registry['function_registry'].items():
            action_function['type'] = 'function'
            openai_action_functions.append(action_function)
        openai_functions = openai_tool_functions + openai_action_functions
        return openai_functions 

    def _create_messages_for_function(self, tool_registry, action_registry, dialogue):
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
        all_functions = self._prepare_openai_functions(tool_registry, action_registry)

        # This prompt is not tuned for performance. 
        function_prompt = (
            "# Instruction\n"
            "You are an assistant in estimating function names and arguments given some dialogues in a video game world.\n"
            "You will need the following information to respond to the user's input. \n"
            "Use the following steps to estimate the necessary function names and arguments. \n"
            "\n"
            "1. Read the dialogue and the target item. \n"
            "2. From the given function information, select the functions that can obtain the information you need. \n"
            "3. Fill in the arguments needed by the function as appropriate. \n"
            "Note: You may select multiple functions or no functions at all. \n"
            "\n"
            "# Additional Information \n"
            "{}\n"
            "# Dialogue\n"
            "The user input for the current turn is as follows. \n"
        )

        # 'target_item' is used to indicate what the user is referring to, such as 'this', 'that', 'the one', etc. 
        additional_info = ""
        if len(dialogue[-1]["target_item"]) > 0:
            additional_info = "In the dialogue, the user may be referring to the following items: \n"
            for info in dialogue[-1]["target_item"]:
                additional_info += f'parameter name: name, value: {info["name"]}\n'
            
        input_messages = [{"role": "system", "content": function_prompt.format(additional_info)}, 
            {'role': 'user', 'content': dialogue[-1]['text']}]

        return input_messages, all_functions
        
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
            "You are an assistant that plays the role of a character in a video game. \n"
            "Use the following character settings and knowledge to create your response.\n"
            "\n"
            "# Character Settings: You should act as the following character. \n"
            "{}\n"
            "\n"
            "# Knowledge\n"
            "There are two parts of knowledge. The first part is the specific knowledge obtained from the function calls. \n"
            "The second part is the general knowledge of all items involved in the dialogue. \n"
            "\n"
            "## Knowledge from Function Calls\n"
            "{}\n"
            "## General Knowledge of All Items\n"
            "{}\n"
            "\n"
            "# Worldview: It describes the setting of the world in the video game. \n"
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
        

        prompt = dialogue_prompt.format(character_setting, function_knowledge, general_knowledge, worldview)
        
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages


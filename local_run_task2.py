from typing import List, Dict
import json
import numpy as np
import npcdataset.parsers
from agents.user_config import UserAgent
import argparse 
from tqdm import tqdm
import os
import time
from function_calls import tool_map, action_map, Executor
from dotenv import load_dotenv
import openai
from bert_score import BERTScorer
import warnings

warnings.filterwarnings(
    "ignore",
    message="The hypothesis contains 0 counts of.*",
    module="nltk.translate.bleu_score"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def evaluate_with_llm(gold, generated):
    prompt = f"""
You are an evaluation assistant.

Gold response (ideal answer):
\"\"\"{gold}\"\"\"  

Generated response (model output):
\"\"\"{generated}\"\"\"  

Evaluate how similar and appropriate the generated response is compared to the gold response.
Give a score from 1 to 5, where:
1 = Very poor (off-topic or incorrect),
3 = Acceptable but not ideal,
5 = Very good (matches tone, information, and style).

Only return the score as a number. No explanation.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        score_text = response.choices[0].message.content.strip()
        return int(score_text)
    except Exception as e:
        print("LLM 평가 오류:", e)
        return -1

scorer = BERTScorer(lang="en", model_type="roberta-large", rescale_with_baseline=True)

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor) -> List[Dict[str, str]]:
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
    parser.add_argument('--save_path', type=str, default='results/task2_responses.json')
    args = parser.parse_args()

    start_time = time.time()
    data_path = 'data/task2_sample.json'
    data_set = load_data(data_path)
    agent = UserAgent()

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    generated_responses = []
    all_generated = []
    all_golds = []
    all_llm_scores = []

    print("🔍 Generating responses and calculating LLM scores...")
    for conv_idx, conversation in tqdm(enumerate(data_set), desc="📁 Conversations"):
        cur_conv_responses = {"id": conversation.id}
        tool_registry = tool_map[conversation.function_list_id]
        action_registry = action_map[conversation.function_list_id]

        for turn_idx, turn in tqdm(
            list(enumerate(conversation.turns)), 
            desc=f"💬 Conversation {conv_idx}", 
            leave=False
        ):
            gold_functions = [
                {"name": fn.name, "parameters": fn.parameters, "return": fn.return_values}
                for fn in turn.gold_functions
            ]
            cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)
            response = get_responses(agent, conversation, turn,
                                     tool_registry, action_registry, cur_turn_exec)
            gold_response = turn.gold_response

            llm_score = evaluate_with_llm(gold_response, response)
            all_llm_scores.append(llm_score)

            print(f"\n[🔹 Turn {turn_idx}]")
            print(f"[🧠 Generated] {response}")
            print(f"[🎯 Gold]      {gold_response}")
            print(f"[⭐ LLM Score] {llm_score}")

            cur_conv_responses[f"turn_{turn_idx}"] = {
                "generated_response": response,
                "gold_response": gold_response,
                "llm_score": llm_score,
                "bertscore_f1": None  # BERTScore placeholder
            }

            all_generated.append(response)
            all_golds.append(gold_response)

        generated_responses.append(cur_conv_responses)

    print("\n📊 Calculating BERTScore for all responses (batch)...")
    P_list, R_list, F1_list = scorer.score(all_generated, all_golds)

    all_bert_f1 = []
    idx = 0
    for conv in generated_responses:
        for key in sorted(k for k in conv if k.startswith("turn_")):
            f1 = F1_list[idx].item()
            conv[key]["bertscore_f1"] = f1
            all_bert_f1.append(f1)
            idx += 1

    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(generated_responses, f, indent=4, ensure_ascii=False)
    print(f"\n✅ All results saved to: {args.save_path}")

    # ✅ 평균 점수 출력
    avg_llm = sum(all_llm_scores) / len(all_llm_scores) if all_llm_scores else 0
    avg_bert = sum(all_bert_f1) / len(all_bert_f1) if all_bert_f1 else 0
    print(f"\n📈 Average LLM Score: {avg_llm:.3f}")
    print(f"📘 Average BERTScore F1: {avg_bert:.4f}")

    print("⏱️ Total time spent:", round(time.time() - start_time, 2), "seconds")


# from typing import List, Dict, Tuple
# import json
# import numpy as np
# import npcdataset.parsers
# from agents.user_config import UserAgent
# import argparse 
# from tqdm import tqdm
# import os
# import time
# from function_calls import tool_map, action_map, Executor
# from dotenv import load_dotenv
# import openai
# from nltk.translate.bleu_score import sentence_bleu
# from bert_score import BERTScorer  # ✅ BERTScore 추가
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import warnings
# # ← optional: ignore only NLTK BLEU warnings
# warnings.filterwarnings(
#     "ignore",
#     message="The hypothesis contains 0 counts of.*",
#     module="nltk.translate.bleu_score"
# )

# # set up one smoothing function instance
# _smooth = SmoothingFunction().method2


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")


# # ✅ 평가 함수 1: GPT-4o-mini로 평가
# def evaluate_with_llm(gold, generated):
#     prompt = f"""
# You are an evaluation assistant.

# Gold response (ideal answer):
# \"\"\"{gold}\"\"\"  

# Generated response (model output):
# \"\"\"{generated}\"\"\"  

# Evaluate how similar and appropriate the generated response is compared to the gold response.
# Give a score from 1 to 5, where:
# 1 = Very poor (off-topic or incorrect),
# 3 = Acceptable but not ideal,
# 5 = Very good (matches tone, information, and style).

# Only return the score as a number. No explanation.
# """
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         score_text = response.choices[0].message.content.strip()
#         return int(score_text)
#     except Exception as e:
#         print("LLM 평가 오류:", e)
#         return -1



# # ✅ 평가 함수 2: BLEU + BERTScore
# scorer = BERTScorer(lang="en", model_type="roberta-large", rescale_with_baseline=True)



# # 1) BLEU만 계산해서 리턴
# def evaluate_with_metrics(gold: str, generated: str) -> dict:
#     # apply smoothing_function to avoid zero scores on short texts
#     bleu = sentence_bleu(
#         [gold.split()],
#         generated.split(),
#         smoothing_function=_smooth
#     )
#     return {"bleu": bleu}


# def load_data(file_path):
#     with open(file_path, "r", encoding="utf-8") as fp:
#         data = json.load(fp)
#     return npcdataset.parsers.parse_conversation_data(data, "test")


# def get_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor) -> List[Dict[str, str]]:
#     dialogue = [
#         {
#             "speaker": msg.speaker,
#             "text": msg.text,
#             "target_item": msg.target_items
#         }
#         for msg in cur_turn.messages
#     ]
#     all_results = agent.generate_functions_and_responses(
#         tool_registry,
#         action_registry,
#         cur_conv.worldview,
#         cur_conv.personas['npc'].to_dict(),
#         cur_conv.roles['npc'],
#         {"general_info": cur_conv.general_knowledge, "knowledge_info": cur_conv.knowledge},
#         cur_conv.state,
#         dialogue,
#         executor
#     )
#     return all_results['final_responses']


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_path', type=str, default='results/task2_responses.json')
#     args = parser.parse_args()

#     start_time = time.time()

#     # ✅ 사용할 데이터 경로 (최종 학습 데이터 기준으로 교체 가능)
#     # data_path = 'data/task2_train.json'

#     data_path = 'data/task2_sample.json'

#     data_set = load_data(data_path)
#     agent = UserAgent()

#     save_directory = os.path.dirname(args.save_path)
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)

            
#     generated_responses = []
#     all_generated = []
#     all_golds = []

#     print("🔍 Generating responses and calculating LLM / BLEU scores...")
#     for conv_idx, conversation in tqdm(enumerate(data_set), desc="📁 Conversations"):
#         cur_conv_responses = {"id": conversation.id}
#         tool_registry = tool_map[conversation.function_list_id]
#         action_registry = action_map[conversation.function_list_id]

#         for turn_idx, turn in tqdm(
#             list(enumerate(conversation.turns)), 
#             desc=f"💬 Conversation {conv_idx}", 
#             leave=False
#         ):
#             gold_functions = [
#                 {"name": fn.name, "parameters": fn.parameters, "return": fn.return_values}
#                 for fn in turn.gold_functions
#             ]
#             cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)
#             response = get_responses(agent, conversation, turn,
#                                     tool_registry, action_registry, cur_turn_exec)
#             gold_response = turn.gold_response

#             # ✅ 점수 계산
#             llm_score = evaluate_with_llm(gold_response, response)
#             bleu_score = evaluate_with_metrics(gold_response, response)["bleu"]

#             # ✅ 디버깅 출력
#             print(f"\n[🔹 Turn {turn_idx}]")
#             print(f"[🧠 Generated] {response}")
#             print(f"[🎯 Gold]      {gold_response}")
#             print(f"[⭐ LLM Score] {llm_score}, [📘 BLEU] {bleu_score:.4f}")

#             cur_conv_responses[f"turn_{turn_idx}"] = {
#                 "generated_response": response,
#                 "gold_response": gold_response,
#                 "llm_score": llm_score,
#                 "bleu": bleu_score,
#                 "bertscore_f1": None  # BERTScore placeholder
#             }

#             all_generated.append(response)
#             all_golds.append(gold_response)

#         generated_responses.append(cur_conv_responses)

#     # ✅ BERTScore 계산
#     print("\n📊 Calculating BERTScore for all responses (batch)...")
#     P_list, R_list, F1_list = scorer.score(all_generated, all_golds)

#     # ✅ BERTScore 결과 삽입
#     idx = 0
#     for conv in generated_responses:
#         for key in sorted(k for k in conv if k.startswith("turn_")):
#             conv[key]["bertscore_f1"] = F1_list[idx].item()
#             idx += 1

#     # ✅ 파일 저장
#     with open(args.save_path, 'w', encoding='utf-8') as f:
#         json.dump(generated_responses, f, indent=4, ensure_ascii=False)
#     print(f"\n✅ All results saved to: {args.save_path}")
#     print("⏱️ Total time spent:", round(time.time() - start_time, 2), "seconds")

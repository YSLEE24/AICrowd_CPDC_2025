# main_tool_embedder.py

import os
import inspect
import importlib.util
import pkgutil
import pickle
from typing import Dict

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 🔹 임베딩 모델

FUNCTION_CALLS_DIR = "function_calls"  # 🔹 폴더 이름 (같은 경로에 있다고 가정)

def load_modules_from_directory(directory: str) -> Dict[str, object]:
    """폴더 내의 모든 .py 모듈을 불러온다."""
    modules = {}
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            filepath = os.path.join(directory, filename)
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[module_name] = module
    return modules


def extract_all_functions_from_modules(modules: Dict[str, object]) -> Dict[str, str]:
    """모든 모듈에서 함수 설명을 추출한다."""
    descriptions = {}
    for mod_name, module in modules.items():
        for name, func in inspect.getmembers(module, inspect.isfunction):
            doc = inspect.getdoc(func) or ""
            sig = inspect.signature(func)
            full_desc = f"{name} {sig}: {doc}"
            descriptions[f"{mod_name}.{name}"] = full_desc
    return descriptions


def embed_and_save(descriptions: Dict[str, str], output_path: str = "tool_embeddings.pkl"):
    """함수 설명을 임베딩하고 pkl로 저장한다."""
    embeddings = {
        name: model.encode(desc)
        for name, desc in descriptions.items()
    }
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"✅ {len(embeddings)}개의 임베딩이 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    print("📦 함수 모듈 로딩 중...")
    modules = load_modules_from_directory(FUNCTION_CALLS_DIR)

    print("🔍 함수 설명 추출 중...")
    descriptions = extract_all_functions_from_modules(modules)

    print("📐 임베딩 + 저장 중...")
    embed_and_save(descriptions)



# from function_calls import tool_map, action_map
# from sentence_transformers import SentenceTransformer
# import pickle

# # ✅ 모든 함수 모듈 가져오기
# all_function_modules = list(tool_map.values()) + list(action_map.values())

# # ✅ 함수 설명 추출
# def extract_tool_descriptions(modules):
#     descriptions = {}
#     for module in modules:
#         registry = module.get('function_registry', {})
#         for name, func in registry.items():
#             desc = func.get('description', '')
#             descriptions[name] = f"{name}: {desc}"
#     return descriptions

# tool_desc = extract_tool_descriptions(all_function_modules)

# # ✅ 임베딩 생성
# model = SentenceTransformer("all-MiniLM-L6-v2")

# tool_embeddings = {
#     name: model.encode(desc)
#     for name, desc in tool_desc.items()
# }

# # ✅ 캐싱 저장
# with open("tool_embeddings.pkl", "wb") as f:
#     pickle.dump(tool_embeddings, f)
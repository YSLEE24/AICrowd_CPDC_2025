import os
import json

### --- 파일 경로 처리 --- ###
rel_path = os.path.join("data", "task1_sample.json")  # 🔧 여기를 수정한 거야
abs_path = os.path.abspath(rel_path)

print("📁 상대경로:", rel_path)
print("📁 절대경로:", abs_path)
print("✅ 파일 존재 여부:", os.path.exists(abs_path))

### --- JSON 파일 로딩 --- ###
try:
    with open(abs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("✅ JSON 파일 로딩 성공")
except Exception as e:
    print("❌ JSON 파일 로딩 실패:", e)

### ================================================
### 3. 각 대화 내의 function_call 확인
### ================================================

all_gold_funcs = {}

for idx, entry in enumerate(data):
    conv_id = str(entry.get("id", idx))  # 없을 경우 idx로 대체
    gold_funcs = []
    print(f"\n📚 Conversation {conv_id}")
    for turn_idx, turn in enumerate(entry.get("dialogue", [])):
        tool_calls = turn.get("tool_calls", [])
        if tool_calls:
            print(f"🛠️ Turn {turn_idx} has tool_calls: {tool_calls}")
        for call in tool_calls:
            func_name = call.get("function", {}).get("name")
            if func_name:
                gold_funcs.append(func_name)
                print(f"🔧 Found function: {func_name}")
    all_gold_funcs[conv_id] = gold_funcs
    print(f"\n📌 Conversation {conv_id} - Gold Functions: {gold_funcs}")

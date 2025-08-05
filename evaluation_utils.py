# evaluation_utils.py

def extract_gold_functions(data):
    gold_funcs = {}
    for entry in data:
        conv_id = str(entry["data_id"])
        func_list = []
        for turn in entry.get("dialogue", []):
            for call in turn.get("gold_functions", []):
                func_name = call.get("name")
                if func_name:
                    func_list.append(func_name)
        gold_funcs[conv_id] = func_list
    return gold_funcs

def extract_predicted_functions(result_data):
    pred_funcs = {}
    for item in result_data:
        conv_id = str(item.get("data_id"))
        funcs = []
        for turn in item.get("outputs", []):
            for call in turn.get("tool_calls", []):
                func_name = call.get("function", {}).get("name")
                if func_name:
                    funcs.append(func_name)
        pred_funcs[conv_id] = funcs
    return pred_funcs

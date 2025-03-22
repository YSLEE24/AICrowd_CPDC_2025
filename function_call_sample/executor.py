import copy

class Executor: 
    """
        A wrapper for function calls. 
        It implements the function calls by calling `execute` in function_calls,
        and records the function call names and args for evaluation purposes. 
    
        Notes: 
            1. This executor is a sample one. It will only check whether the output matches the gold functions. 
               If it matches, we will return the gold return values. 
               It it does not match, it will return nothing. 
               However, in real evaluations, the executor will return adequate values even though it is not an exact match with gold functions. 
            2. Please do not try to tamper with attributes in the Executor. Doing so will lead to errors. 
    """
    def __init__(self, tool_registry, action_registry, gold_functions):
        self.function_call_stats = []
        self.tool_registry = tool_registry
        self.action_registry = action_registry
        self.gold_functions = gold_functions
    
    def execute(self, function_list): 
        """
            Execute the list of functions by checking the gold functions.
            It will also record the function call names and args (for evaluation purposes). 
        """
        copy_functions = copy.deepcopy(function_list)
        for func_item in copy_functions:
            self.function_call_stats.append(copy.deepcopy(func_item))

            # if it matches a gold function
            gold_func_index = self.check_exact_match_gold(func_item)
            if gold_func_index != -1:
                # matches, we return the gold return value
                func_item['return'] = copy.deepcopy(self.gold_functions[gold_func_index]['return'])
            else:
                func_item['return'] = []
            
        return copy_functions
    
    def check_exact_match_gold(self, func_item):
        """
            Checks if the func_item matches any of the gold functions.
            If yes, return the matching index. 
            If not, return -1.  
        """
        for i, gold_function in enumerate(self.gold_functions):
            if gold_function['name'] == func_item['name']:
                # function call name match
                gold_function_param_name_list = list(gold_function['parameters'].keys())
                gold_function_param_name_list.sort()
                generated_func_param_name_list = list(func_item['parameters'].keys())
                generated_func_param_name_list.sort() 
                if gold_function_param_name_list == generated_func_param_name_list:
                    # param names match
                    gold_function_param_value_list = [gold_function["parameters"][param_name] for param_name in gold_function_param_name_list]
                    generated_func_param_value_list = [func_item["parameters"][param_name] for param_name in generated_func_param_name_list]
                    if gold_function_param_value_list == generated_func_param_value_list:
                        return i 
        return -1
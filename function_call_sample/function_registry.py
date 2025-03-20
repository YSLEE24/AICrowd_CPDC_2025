class FunctionRegistry: 
    def __init__(self, func_dict={}):
        self.function_registry = func_dict
        self.knowledge = {}
    
    def get_function(self, func_name):
        if func_name in self.function_registry:
            return self.function_registry[func_name]
        else:
            return False
    
    def has_function(self, func_name):
        return func_name in self.function_registry

    
    def all_functions(self):
        return self.function_registry.keys()
    
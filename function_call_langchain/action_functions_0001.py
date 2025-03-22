from langchain.tools import tool
from typing import List, Dict, Tuple


@tool
def sell(item_names: List[str]) -> None:
    """
    Sell the specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name: List[str]
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    None    
    """

    pass

@tool 
def equip(item_name: str) -> None:
    """
    Equip the specified weapon (e.g. Avis Wind, Short Sword, etc.).
    
    Parameters:
    ----------
    item_name: str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

all_functions = [sell, equip]
action_functions_0001 = {'function_registry': {
    f.name: {
        'name': f.name, 
        'description': f.description,
        'args': f.args
    }
    for f in all_functions
}}



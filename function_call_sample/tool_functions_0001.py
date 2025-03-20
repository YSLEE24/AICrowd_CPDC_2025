from typing import List, Dict
import function_calls
from .function_registry import FunctionRegistry 
from langchain.tools import tool



def search_item(item_name: str, item_price: str, item_type: str, item_attack: str, item_description: str,
                item_name_operator: str, item_price_operator: str, item_type_operator: str, item_attack_operator: str) -> List[Dict[str, str]]:
    """
    Search for weapons based on specified criteria,
    such as price(e.g. 10G, 500G,  etc.), type (e.g. spear, bow, etc.), attack level (e.g. 10, 100, etc.), and specific features (e.g.  beginner-friendly, lightweight, etc.).
    Returns a list of weapon names along with the reasons for the selection. It returns 'many' when there are multiple applicable items, and 'n/a' when there are none.

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation. Multiple weapon names can be set (e.g. Avis Wind | Short Sword).

    item_price : str
        Specified price (e.g. 10G, 500G, etc.). Uses the price mentioned in the conversation.

    item_type : str
        Specified weapon type (e.g. spear, bow, etc.).
        Recognizes the weapon type mentioned in the conversation and applies the corresponding weapon type from the knowledge base,
        using one of the following: axe, blunt weapon, bow, sword, double-handed sword, single-handed sword, spear, whip.

    item_attack : str
        Specified weapon attack level (e.g. 10, 100, etc.). Uses the attack level of the weapon mentioned in the conversation.

    item_description : str
        Specified weapon characteristics (e.g. beginner-friendly, light, etc.). Uses the characteristics of the weapon mentioned in the conversation.

    item_name_operator : str
        Specified weapon characteristics (e.g. beginner-friendly, light, etc.). Uses the characteristics of the weapon mentioned in the conversation.

    item_price_operator : str
        Modifier for comparison and exclusion used to describe the price specified by item_price.
        The modifier can be one of the following: no limit, or more, or less, highest, high, average, low, lowest, other than.

    item_type_operator : str
        Exclusion modifier used with the weapon type specified by item_type. Uses 'other than' as the modifier.

    item_attack_operator : str
        Modifier for comparison and exclusion used to describe the weapon attack level specified by item_attack.
        The modifier can be one of the following: no limit, or more, or less, highest, high, average, low, lowest, other than.

    Returns:
    -------
    List[Dict[str, str]]
        A list of weapon names along with the reasons for the selection.

    """
    
    function = "search_item"
    args = {
        "item_name": item_name,
        "item_price": item_price,
        "item_type": item_type,
        "item_attack": item_attack,
        "item_description": item_description,
        "item_name_operator": item_name_operator,
        "item_price_operator": item_price_operator,
        "item_type_operator": item_type_operator,
        "item_attack_operator": item_attack_operator
    }
    ret = function_calls.execute(function, args)
    return ret


def check_inventory(item_name: str) -> List[Dict[str, str]]:
    """
    Check the availability for a specified weapon(e.g. Avis Wind, Short Sword, etc.)

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the abailable weapon (e.g. Avis Wind, Short Sword, etc.).

    """
    return []


def check_basic_info(item_name: str) -> List[Dict[str, str]]:
    """
    Check the price, type, attack level, and basic information of a specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs basic information about the specified weapon (e.g. Avis Wind, Short Sword, etc.).

    """
    
    function = "check_basic_info"
    args = {"item_name": item_name}
    ret = function_calls.execute(function, args)
    return ret


def check_price(item_name: str) -> List[Dict[str, str]]:
    """
    Check the price of a specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation. 

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the price of the specified weapon (e.g. Avis Wind, Short Sword, etc.)

    """

    function = "check_price"
    args = {"item_name": item_name}
    ret = function_calls.execute(function, args)
    return ret


def check_type(item_name: str) -> List[Dict[str, str]]:
    """
    Check the type of a specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the type of the specified weapon (e.g. Avis Wind, Short Sword, etc.)

    """

    function = "check_type"
    args = {"item_name": item_name}
    ret = function_calls.execute(function, args)
    return ret


def check_attack(item_name: str) -> List[Dict[str, str]]:
    """
    Check the attack level of a specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the attack level of the specified weapon (e.g. Avis Wind, Short Sword, etc.)
    """

    function = "check_attack"
    args = {"item_name": item_name}
    ret = function_calls.execute(function, args)
    return ret


def check_description(item_name: str) -> List[Dict[str, str]]:
    """
    Check the basic information and additional detailed information of the specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name : str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the basic information and additional detailed information of the specified weapon (e.g. Avis Wind, Short Sword, etc.)
    """

    function = "check_description"
    args = {"item_name": item_name}
    ret = function_calls.execute(function, args)
    return ret


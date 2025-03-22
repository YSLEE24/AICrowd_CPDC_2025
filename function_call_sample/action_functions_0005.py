from langchain.tools import tool

@tool 
def select_request_confirm(quest_name: str) -> None:
    """
    Confirm whether to select the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def select(quest_name: str) -> None:
    """
    Select the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
 
    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def start(quest_name: str) -> None:
    """
    Start the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

all_functions = [select_request_confirm, select, start]
action_functions_0005 = {'function_registry': {
    f.name: {
        'name': f.name, 
        'description': f.description,
        'args': f.args
    }
    for f in all_functions
}}
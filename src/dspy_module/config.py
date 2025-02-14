"""
config.py module handles the configuration for the dspy library.
"""

import json
import dspy

def load_config(filepath):
    """
    Load configuration from a specified JSON file.

    Args:
        filepath (str): The path to the configuration file.

    Returns:
        dspy.LM: The configured language model.
    """
    with open(filepath, 'r') as file:
        config = json.load(file)
    lm = dspy.LM(config['model'], api_base=config['api_base'], api_key=config['api_key'])
    dspy.configure(lm=lm)
    return lm

lm = load_config('src/dspy_module/config.json')

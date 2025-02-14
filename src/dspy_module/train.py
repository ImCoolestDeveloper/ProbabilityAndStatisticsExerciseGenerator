# src/dspy_module/train.py

"""
train.py module handles the training data loading and preparation for the Exercise class.
"""

import dspy

def load_trainset(filepath):
    """
    Load training data from a specified plain text file.

    Args:
        filepath (str): The path to the training data file.

    Returns:
        list: A list of dspy.Example objects representing the training data.
    """
    trainset = []
    with open(filepath, 'r') as file:
        data = file.read().split('!!!')
        for i in range(0, len(data) - 2, 3):
            example = {
                "type": data[i].strip(),
                "text": data[i+1].strip(),
                "correct_answer": data[i+2].strip()
            }
            trainset.append(dspy.Example(**example))
    return trainset

if __name__ == "__main__":
    trainset = load_trainset('src/dspy_module/train_data.txt')
    for example in trainset:
        print(example)

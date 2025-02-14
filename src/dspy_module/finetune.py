# src/dspy_module/finetune.py

"""
finetune.py module handles the fine-tuning process for the Exercise class.
"""

import dspy
from .train import load_trainset
from .gen_exercise_module import gen_exercise_module

def fine_tune_model(filepath):
    """
    Fine-tune the Exercise model with the training data.

    Args:
        filepath (str): The path to the training data file.

    Returns:
        dspy.Predict: The fine-tuned Exercise model.
    """
    trainset = load_trainset(filepath)
    optimizedGenExercise = dspy.BootstrapFinetune(
        metric=(lambda x, y, trace=None: x.label == y.label),
        num_threads=24
    )
    optimizedGenExercise = optimizedGenExercise.compile(gen_exercise_module, trainset=trainset)
    return optimizedGenExercise

if __name__ == "__main__":
    optimizedGenExercise = fine_tune_model('src/dspy_module/train_data.txt')
    print("Model fine-tuned successfully!")

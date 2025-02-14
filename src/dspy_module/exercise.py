# src/dspy_module/exercise.py

"""
exercise.py module contains the Exercise class and related functions for generating exercises.
"""

import dspy

class Exercise(dspy.Signature):
    """
    The Exercise class represents an exercise with type, text, and correct answer.

    Attributes:
        type (str): The type of exercise.
        text (str): The exercise text.
        correct_answer (str): The correct answer for the exercise.
    """
    
    type: str = dspy.InputField()
    text: str = dspy.OutputField()
    correct_answer: str = dspy.OutputField()

if __name__ == "__main__":
    exercise = Exercise(type="Example", text="This is an example exercise", correct_answer="Example answer")
    print(exercise)

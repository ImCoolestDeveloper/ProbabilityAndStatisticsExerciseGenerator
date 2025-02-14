# src/dspy_module/gen_exercise_module.py

"""
gen_exercise_module.py module contains the prediction functions for the Exercise class.
"""

import dspy
from .exercise import Exercise

gen_exercise_module = dspy.Predict(Exercise)

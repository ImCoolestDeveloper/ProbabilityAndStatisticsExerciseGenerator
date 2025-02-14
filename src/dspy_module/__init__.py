"""
dspy_module package initializer.
"""

import dspy
from .exercise import Exercise
from .gen_exercise_module import gen_exercise_module
from .finetune import fine_tune_model
from .train import load_trainset
from .config import lm


dspy.configure(lm=lm)

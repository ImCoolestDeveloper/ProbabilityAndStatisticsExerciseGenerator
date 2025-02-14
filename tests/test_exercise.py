# tests/test_exercise.py

import unittest
from dspy_module.train import load_trainset
from dspy_module.gen_exercise_module import gen_exercise_module
from dspy_module.finetune import fine_tune_model

class TestExercise(unittest.TestCase):
    
    def test_load_trainset(self):
        trainset = load_trainset('src/dspy_module/train_data.txt')
        self.assertEqual(len(trainset), 2)  # Assuming there are 2 exercises in the train data

    def test_gen_exercise_module(self):
        exercise = gen_exercise_module(exercise_type="Create a probability exercise")
        self.assertIsInstance(exercise, dict)
        self.assertIn("text", exercise)
    
    def test_optimizedGenExercise(self):
        optimizedGenExercise = fine_tune_model('src/dspy_module/train_data.txt')
        exercise = optimizedGenExercise(exercise_type="Create a probability exercise")
        self.assertIsInstance(exercise, dict)
        self.assertIn("text", exercise)

if __name__ == '__main__':
    unittest.main()

# src/main.py

from dspy_module.finetune import fine_tune_model

def main():
    """
    Main
    """
    optimizedGenExercise = fine_tune_model('src/dspy_module/train_data.txt')
    result = optimizedGenExercise(exercise_type="Create a probability exercise")
    print(result)

if __name__ == "__main__":
    main()

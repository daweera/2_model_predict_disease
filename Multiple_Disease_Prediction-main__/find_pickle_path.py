import os
import pickle

current_directory = os.getcwd()
print("Current Directory:", current_directory)

model_path = os.path.join(
    current_directory,
    "Multiple_Disease_Prediction-main__",
    "saved_models",
    "diabetes.pkl",
)


if os.path.exists(model_path):
    print("Found the file:", model_path)
else:
    print("File not found:", model_path)


with open(model_path, "rb") as file:
    diabetes_model = pickle.load(file)

import json
import requests

print("Choose the model you want to test:")
print(" - Random Forest: rf")
print(" - Logistic Regression: logreg")
print(" - Neural Network: nn\n")

model = input("Enter your choice: ").strip().lower()

if model == "rf":
    url = "http://localhost:8000/predict/randomforest"
    input_data = {
        "radius_mean": 14.0,
        "texture_mean": 20.0,
        "perimeter_mean": 90.0,
        "area_mean": 880.0,
        "smoothness_mean": 0.1,
        "compactness_mean": 0.2,
        "concavity_mean": 0.2,
        "concave_points_mean": 0.1,
        "symmetry_mean": 0.2,
        "fractal_dimension_mean": 0.06,
        "radius_se": 0.5,
        "texture_se": 1.0,
        "perimeter_se": 2.0,
        "area_se": 20.0,
        "smoothness_se": 0.01,
        "compactness_se": 0.02,
        "concavity_se": 0.03,
        "concave_points_se": 0.01,
        "symmetry_se": 0.02,
        "fractal_dimension_se": 0.003,
        "radius_worst": 17.0,
        "texture_worst": 25.0,
        "perimeter_worst": 110.0,
        "area_worst": 1000.0,
        "smoothness_worst": 0.15,
        "compactness_worst": 0.25,
        "concavity_worst": 0.3,
        "concave_points_worst": 0.14,
        "symmetry_worst": 0.3,
        "fractal_dimension_worst": 0.08
    }

elif model == "logreg":
    url = "http://localhost:8000/predict/logistic"
    input_data = {
        "radius_mean": 14.0,
        "texture_mean": 20.0,
        "compactness_worst": 0.25,
        "concave_points_worst": 0.15,
        "area_worst": 880.0
    }

elif model == "nn":
    url = "http://localhost:8000/predict/neuralnet"
    input_data = {
        "radius_mean": 14.0,
        "texture_mean": 20.0,
        "perimeter_mean": 90.0,
        "area_mean": 880.0,
        "smoothness_mean": 0.1,
        "compactness_mean": 0.2,
        "concavity_mean": 0.2,
        "concave_points_mean": 0.1,
        "symmetry_mean": 0.2,
        "fractal_dimension_mean": 0.06,
        "radius_se": 0.5,
        "texture_se": 1.0,
        "perimeter_se": 2.0,
        "area_se": 20.0,
        "smoothness_se": 0.01,
        "compactness_se": 0.02,
        "concavity_se": 0.03,
        "concave_points_se": 0.01,
        "symmetry_se": 0.02,
        "fractal_dimension_se": 0.003,
        "radius_worst": 17.0,
        "texture_worst": 25.0,
        "perimeter_worst": 110.0,
        "area_worst": 1000.0,
        "smoothness_worst": 0.15,
        "compactness_worst": 0.25,
        "concavity_worst": 0.3,
        "concave_points_worst": 0.14,
        "symmetry_worst": 0.3,
        "fractal_dimension_worst": 0.08
    }

else:
    print("Invalid model choice. Please choose 'rf', 'logreg', or 'nn'.")
    exit()

headers = {"Content-Type": "application/json"}
input_json = json.dumps(input_data)

response = requests.post(url, data=input_json, headers=headers)

print("\nStatus Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2))

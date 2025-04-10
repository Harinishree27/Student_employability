import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Try loading the trained scaler and model, otherwise train and save them
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
   
    with open("logistic_regression.pkl", "rb") as model_file:
        model = pickle.load(model_file)

except FileNotFoundError:
    print("Model or scaler file not found. Training the model now...")
   
    # Load dataset
    df = pd.read_csv("Student-Employability-Datasets.csv")
   
    # Feature selection
    X = df.iloc[:, 1:-2].values
    y = (df["CLASS"] == "Employable").astype(int)  # Convert labels to binary (0 or 1)
   
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
   
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully with accuracy: {accuracy:.2f}")
   
    # Save the scaler and model
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open("logistic_regression.pkl", "wb") as model_file:
        pickle.dump(model, model_file)


def predict_employability(name, general_appearance, manner_of_speaking, physical_condition,
                           mental_alertness, self_confidence, ability_to_present_ideas,
                           communication_skills):
    # Create input array
    input_data = np.array([[general_appearance, manner_of_speaking, physical_condition,
                             mental_alertness, self_confidence, ability_to_present_ideas,
                             communication_skills]])
   
    # Scale input data
    input_scaled = scaler.transform(input_data)
   
    # Predict
    prediction = model.predict(input_scaled)
   
    # Convert prediction to label with animation
    if prediction[0] == 1:
        result = f"{name} is Employable 🎉🎊🎆✨ (Crackers bursting animation, excited and surprised expressions)"
    else:
        result = f"{name} is Less Employable 😞😢 (Sad and sorrow expressions)"
   
    return result

# Define input interface
inputs = [
    gr.Textbox(label="Name"),
    gr.Slider(1, 5, step=1, label="General Appearance"),
    gr.Slider(1, 5, step=1, label="Manner of Speaking"),
    gr.Slider(1, 5, step=1, label="Physical Condition"),
    gr.Slider(1, 5, step=1, label="Mental Alertness"),
    gr.Slider(1, 5, step=1, label="Self Confidence"),
    gr.Slider(1, 5, step=1, label="Ability to Present Ideas"),
    gr.Slider(1, 5, step=1, label="Communication Skills"),
]

# Define output
output = gr.Textbox(label="Employability Prediction")

# Create Gradio interface
app = gr.Interface(fn=predict_employability, inputs=inputs, outputs=output)

# Launch the app 
app.launch()

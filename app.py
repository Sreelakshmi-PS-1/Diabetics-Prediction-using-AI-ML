import gradio as gr
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):

    input_data = np.array([
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]).reshape(1, -1)

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    return "🛑 Diabetics Detected" if prediction[0] == 1 else "✅ No Diabetes Detected"


custom_css = """
body {
    background-color: #f4faf7;
}

.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}

h1, h3 {
    color: #2f4f4f;
}

button {
    background-color: #7fbfa4 !important;
    color: #ffffff !important;
    border-radius: 8px;
    font-weight: 600;
}

button:hover {
    background-color: #6aad96 !important;
}

textarea, input {
    border-radius: 6px !important;
    border: 1px solid #c7e3d4 !important;
}

label {
    color: #355f5b;
    font-weight: 500;
}
"""

with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("""
    # 🧠 AI-Based Diabetes Prediction System  
    ### Calm & Eye-Pleasant Medical Interface

    Enter patient medical details to check diabetes status.
    """)

    with gr.Row():
        with gr.Column():
            Pregnancies = gr.Number(label="🤰 Pregnancies", value=0)
            Glucose = gr.Number(label="🩸 Glucose Level", value=120)
            BloodPressure = gr.Number(label="💓 Blood Pressure", value=70)
            SkinThickness = gr.Number(label="📏 Skin Thickness", value=20)

        with gr.Column():
            Insulin = gr.Number(label="💉 Insulin Level", value=85)
            BMI = gr.Number(label="⚖️ BMI", value=28.5)
            DiabetesPedigreeFunction = gr.Number(
                label="🧬 Diabetes Pedigree Function", value=0.5
            )
            Age = gr.Number(label="🎂 Age", value=32)

    predict_btn = gr.Button("🔍 Predict Diabetes")

    output = gr.Textbox(
        label="🧪 Prediction Result",
        interactive=False
    )

    predict_btn.click(
        predict_diabetes,
        inputs=[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ],
        outputs=output
    )

    gr.Markdown("""
    ---
    🔬 *For educational use only. Not a substitute for medical diagnosis.*
    """)

demo.launch()
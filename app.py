import streamlit as st
import joblib

# ------------------ Load Models Safely ------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model.pkl")
        tfidf = joblib.load("tfidf.pkl")
        encoder = joblib.load("encoder.pkl")
        return model, tfidf, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, tfidf, encoder = load_models()

# ------------------ App UI ------------------
st.set_page_config(page_title="Student Level Predictor", page_icon="🎓")

st.title("🎓 Student Learning Level Predictor")
st.write("Enter student description to predict learning level")

# Example suggestions
example = st.selectbox(
    "Try an example:",
    ["", "weak in basics", "average understanding", "strong problem solving"]
)

user_input = st.text_area(
    "Student Description",
    value=example
)

# ------------------ Prediction Function ------------------
def predict_level(text):
    vector = tfidf.transform([text])
    pred = model.predict(vector)

    # Convert to original label
    label = encoder.inverse_transform(pred.reshape(-1, 1))[0][0]
    return label

# ------------------ Predict Button ------------------
if st.button("Predict"):
    if model is None:
        st.stop()

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a description")
    else:
        try:
            result = predict_level(user_input)

            # Display result with styling
            if result == "Beginner":
                st.error(f"📉 Predicted Level: {result}")
                st.write("👉 Focus on basics and practice simple problems.")

            elif result == "Intermediate":
                st.warning(f"📊 Predicted Level: {result}")
                st.write("👉 Improve problem-solving and practice regularly.")

            else:
                st.success(f"🚀 Predicted Level: {result}")
                st.write("👉 Try advanced problems and real-world projects.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ------------------ Footer ------------------
st.write("---")
st.caption("Built using Machine Learning + NLP (TF-IDF + Classification)")
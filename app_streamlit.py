import streamlit as st
import numpy as np
from model import load_model, predict_species

st.title("🌸 Iris Species Prediction")
st.markdown("A simple ML app to classify **Iris species** based on sepal & petal measurements.")

# Load the model
model = load_model()

st.header("🌿 Plant Features")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sepal Characteristics")
    sepal_l = st.slider("Sepal Length (cm)", 1.0, 8.0, 5.1, 0.1)
    sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.5, 0.1)

with col2:
    st.subheader("Petal Characteristics")
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

st.write("")

# Predict button
if st.button("🔮 Predict Iris Type"):
    features = [sepal_l, sepal_w, petal_l, petal_w]
    result = predict_species(model, features)
    st.success(f"🌼 Predicted Species: **{result}**")


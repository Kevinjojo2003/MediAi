import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import os
from utils import (
    process_report,
    generate_lab_report_graph,
    medical_chatbot,
    analyze_medical_image
)

# Ensure the uploads directory exists
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Set Page Config
st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Sidebar Navigation
st.sidebar.title("Medical AI Analyzer")
menu_option = st.sidebar.radio(
    "Navigation",
    ["Medical Report Analysis", "Medical Image Analysis", "Medical Chatbot"]
)

# Medical Report Analysis
if menu_option == "Medical Report Analysis":
    st.title("Medical Report Analyzer")
    st.write("Upload a medical report (PDF or image) to analyze its contents.")

    uploaded_file = st.file_uploader("Upload a medical report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # File Size Check (Prevent Large Uploads)
        if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10 MB limit
            st.error("File too large. Please upload a file under 10MB.")
        else:
            st.write("Analyzing report...")
            result = process_report(file_path)

            if isinstance(result, dict):
                st.subheader("Hugging Face Medical NLP Analysis")
                st.json(result["huggingface_analysis"])

                st.subheader("Gemini AI Medical Insights")
                if "⚠️" in result["gemini_analysis"]:
                    st.warning(result["gemini_analysis"])
                else:
                    st.write(result["gemini_analysis"])

            else:
                st.warning(result)

            if st.button("Generate Lab Report Graph"):
                example_data = {
                    "Hemoglobin": (13.5, "g/dL", (12.0, 16.0)),
                    "WBC Count": (7500, "cells/uL", (4000, 11000))
                }
                graph_path = generate_lab_report_graph(example_data)
                st.image(graph_path, caption="Lab Report Graph", use_column_width=True)

# Medical Image Analysis
elif menu_option == "Medical Image Analysis":
    st.title("Medical Image Analyzer")
    st.write("Upload an X-ray, MRI, or CT Scan for AI-powered analysis.")

    uploaded_image = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        file_path = os.path.join(UPLOADS_DIR, uploaded_image.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(file_path, caption="Uploaded Medical Image", use_column_width=True)

        if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10 MB limit
            st.error("File too large. Please upload a file under 10MB.")
        else:
            st.write("Processing image...")
            image_analysis = analyze_medical_image(file_path)

            if isinstance(image_analysis, str):
                st.warning(image_analysis)
            else:
                st.subheader("Medical Image Segmentation Result")
                st.image(image_analysis, caption="Segmented Medical Image", use_column_width=True)

# Medical Chatbot
elif menu_option == "Medical Chatbot":
    st.title("Medical Chatbot")
    st.write("Ask the AI about medical conditions, symptoms, or reports.")

    user_query = st.text_area("Ask a question about health and medicine:", height=100)

    if st.button("Ask AI"):
        if user_query:
            response = medical_chatbot(user_query)
            st.subheader("AI Response")
            if "⚠️" in response:
                st.warning(response)
            else:
                st.write(response)
        else:
            st.warning("Please enter a question.")

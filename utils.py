import os
import pdfplumber
import pytesseract
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import google.generativeai as genai
from segment_anything import sam_model_registry, SamPredictor
from config import HUGGINGFACE_TOKEN, GEMINI_API_KEY

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Tesseract-OCR path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model paths
MODEL_DIR = "C:/Users/KEVIN JOJO/Desktop/workcohol/Medical-Report-Analyzer/models/"
BIOBERT_MODEL = os.path.join(MODEL_DIR, "biobert")
LAYOUTLM_MODEL = os.path.join(MODEL_DIR, "layoutlmv3")
SAM_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")

# Load BioBERT model for NLP analysis
try:
    biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    biobert_model = AutoModelForSequenceClassification.from_pretrained(BIOBERT_MODEL)
    nlp_pipeline = pipeline("text-classification", model=biobert_model, tokenizer=biobert_tokenizer)
except Exception as e:
    print(f"Error loading BioBERT model: {e}")
    nlp_pipeline = None

# Load SAM model for image segmentation
try:
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)
except Exception as e:
    print(f"Error loading SAM model: {e}")
    sam_predictor = None

# Prompt directory
PROMPT_DIR = "C:/Users/KEVIN JOJO/Desktop/workcohol/Medical-Report-Analyzer/prompts/"

def load_prompt(file_name):
    """Load a prompt text file."""
    prompt_path = os.path.join(PROMPT_DIR, file_name)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def analyze_medical_text(text):
    """Run BioBERT-based classification on medical text."""
    if nlp_pipeline:
        try:
            return nlp_pipeline(text[:512])  # Limit to 512 tokens
        except Exception as e:
            return f"Error analyzing medical text: {e}"
    return "BioBERT model not available."

def analyze_medical_image(image_path):
    """Segment medical images using SAM."""
    if sam_predictor is None:
        return "SAM model not available."

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)

    height, width, _ = image.shape
    input_point = np.array([[width // 2, height // 2]])  # Center point
    input_label = np.array([1])

    masks, _, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    output_path = "assets/segmented_image.png"
    mask_image = (masks[0] * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)

    return output_path

def analyze_with_gemini(text):
    """Use Gemini model to analyze medical text."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = load_prompt("text_analysis.txt")
        response = model.generate_content(f"{prompt}\n{text}")
        return response.text
    except Exception as e:
        return f"Error in Gemini analysis: {e}"

def generate_lab_report_graph(data, save_path="assets/lab_report_graph.png"):
    """
    Generate a graph for lab values.
    Example format:
    {
        "Hemoglobin": (13.5, "g/dL", (12.0, 16.0)),
        "WBC Count": (7500, "cells/uL", (4000, 11000))
    }
    """
    labels, values, units, normal_ranges = [], [], [], []

    for test, (value, unit, normal_range) in data.items():
        labels.append(test)
        values.append(value)
        units.append(unit)
        normal_ranges.append(normal_range)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color='skyblue')

    for i, (low, high) in enumerate(normal_ranges):
        ax.hlines([low, high], i - 0.4, i + 0.4, colors='red', linestyles='dashed')

    ax.set_ylabel("Values")
    ax.set_title("Laboratory Test Report")
    plt.xticks(rotation=45, ha="right")

    plt.savefig(save_path)
    return save_path

def process_report(file_path):
    """Process a PDF or image report and analyze it."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path)
    else:
        return "Unsupported file format."

    if text:
        hf_analysis = analyze_medical_text(text)
        gemini_analysis = analyze_with_gemini(text)
        return {
            "huggingface_analysis": hf_analysis,
            "gemini_analysis": gemini_analysis
        }
    return "No text extracted from the document."

def medical_chatbot(user_input):
    """Use Gemini to answer health-related queries."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = load_prompt("chat_prompt.txt")
        response = model.generate_content(f"{prompt}\nUser: {user_input}\nAI:")
        return response.text
    except Exception as e:
        return f"Error in chatbot response: {e}"

Medical Report Analyzer
AI-powered tool for analyzing medical reports, segmenting medical images, and interacting with a healthcare chatbot.

Live Demo: https://mediai.streamlit.app

Key Features
Medical Report Analysis
Upload PDF or image-based medical reports and extract insights using advanced NLP models.

AI-Powered Text Understanding
Uses Hugging Face's BioBERT and Gemini AI to analyze and interpret medical terminology.

Medical Image Segmentation
Automatically segments and highlights relevant areas in X-rays, MRIs, and CT scans using the SAM model.

Interactive Medical Chatbot
Ask questions about symptoms, conditions, and treatments. The chatbot provides AI-generated responses.

Lab Report Visualization
Automatically generates bar graphs to visualize lab test values and compare them against normal ranges.


Medical-Report-Analyzer/
│
├── uploads/               # Stores uploaded files
├── app.py                 # Main Streamlit application
├── utils.py               # NLP, OCR, image segmentation, chatbot logic
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # File exclusions for Git

How to Use
Report Analysis
Upload a medical report (PDF or image).

The system extracts text using OCR or PDF parsers.

NLP models analyze the content and display medical insights.

Optional: Generate graphs for lab report values.

Image Analysis
Upload an X-ray, MRI, or CT image.

The image is processed and segmented using the SAM model.

View the output with highlighted medical regions.

Medical Chatbot
Ask a question (e.g., "What are the symptoms of asthma?").

The AI provides an evidence-based response using Gemini.


## Configuration
API keys are read from environment variables, never from source code. Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
# GEMINI_API_KEY=...
# HUGGINGFACE_TOKEN=...
pip install -r requirements.txt
streamlit run app.py
```
`uploads/` is created at runtime and is git-ignored. Do not upload real patient data to a shared deployment. This tool is for education and is not medical advice.

## Deployment notes
- `packages.txt` installs Tesseract for Streamlit Cloud. Add `GEMINI_API_KEY` and `HUGGINGFACE_TOKEN` under **Settings > Secrets**.
- The optional image-segmentation step needs the SAM checkpoint `sam_vit_h_4b8939.pth` (about 2.4 GB) in `models/` or at `MODEL_DIR`. It is not in this repo. Without it, the app still runs and skips segmentation.
- The full dependency set (PyTorch, Transformers, SAM) is heavy. Streamlit Community Cloud's free tier has about 1 GB of memory, so a Hugging Face Space or the included `Dockerfile` on a larger host is a better fit.

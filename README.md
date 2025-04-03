# **🩺 Medical Report Analyzer**  
🚀 AI-powered medical report and image analysis tool with chatbot support.
To see the Live Demo https://mediai.streamlit.app/
---

## **📌 Features**  
✅ **Medical Report Analysis** – Extracts and analyzes text from PDFs & images.  
✅ **AI-Powered NLP** – Uses Hugging Face models for medical text analysis.  
✅ **Medical Image Segmentation** – Analyzes X-rays, MRIs, and CT scans with AI.  
✅ **Interactive Chatbot** – Provides medical insights using AI models.  
✅ **Graph Generation** – Visualizes lab results with automated graphs.  

---

## **📂 Project Structure**  

Medical-Report-Analyzer/
│── uploads/                 # Directory for uploaded files
│── app.py                   # Main Streamlit application
│── utils.py                  # Utility functions (NLP, OCR, image processing)
│── requirements.txt          # Dependencies for the project
│── README.md                 # Project documentation (this file)
└── .gitignore                # Ignore unnecessary files


---

## **🚀 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Medical-Report-Analyzer.git
cd Medical-Report-Analyzer

```
### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

### **3️⃣ Install Dependencies**  

pip install -r requirements.txt

### **4️⃣ Run the Application

streamlit run app.py

📊 Usage Guide
📄 Medical Report Analysis
1️⃣ Upload a PDF or Image of a medical report.
2️⃣ AI extracts and analyzes the report using Hugging Face models.
3️⃣ View the medical insights and lab result graphs.

🖼 Medical Image Analysis
1️⃣ Upload an X-ray, MRI, or CT scan image.
2️⃣ AI processes and segments the medical image.
3️⃣ View the segmented medical image.

💬 Medical Chatbot
1️⃣ Type a medical question (e.g., "What are the symptoms of diabetes?").
2️⃣ AI provides relevant medical insights.

📌 Technologies Used
Python – Backend logic

Streamlit – UI & frontend

LangChain – AI-powered text processing

Hugging Face Transformers – NLP & medical text analysis

Segment Anything Model (SAM) – Medical image segmentation

Google Gemini AI – Chatbot & advanced medical analysis

🔧 Future Improvements
✅ Add support for DICOM images
✅ Improve AI chatbot for diagnosis
✅ Implement database support for saving reports

🤝 Contributing
🔹 Fork the repo & submit pull requests.
🔹 Report bugs & request features via Issues.

📜 License
This project is open-source under the MIT License.




import streamlit as st
import PyPDF2
from groq import Groq
import base64
from PIL import Image
import io

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="MedReport AI",
    page_icon="🏥",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedReport AI</h1>
    <p>Understand your medical report simple in own language</p>
    <p>Powered by Groq AI ⚡</p>
</div>
""", unsafe_allow_html=True)

# ── Configure Groq ─────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Supported Languages ────────────────────────────────────
LANGUAGES = {
    "🇬🇧 English": "English",
    "🇱🇰 Sinhala (සිංහල)": "Sinhala",
    "🇱🇰 Tamil (தமிழ்)": "Tamil",
    "🇸🇦 Arabic (عربي)": "Arabic",
    "🇮🇳 Hindi (हिंदी)": "Hindi",
    "🇫🇷 French (Français)": "French",
    "🇩🇪 German (Deutsch)": "German",
    "🇨🇳 Chinese (中文)": "Chinese",
    "🇯🇵 Japanese (日本語)": "Japanese",
    "🇪🇸 Spanish (Español)": "Spanish",
    "🇵🇹 Portuguese": "Portuguese",
    "🇮🇩 Malay (Bahasa)": "Malay",
}

# ── Functions ──────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_image(image_file):
    try:
        # Read image and convert to base64
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Get image format
        image = Image.open(io.BytesIO(image_bytes))
        fmt = image.format.lower()
        if fmt == "jpg":
            fmt = "jpeg"
        media_type = f"image/{fmt}"

        # Use Groq vision model to read image
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """This is a medical report image. 
                            Please extract ALL text from this image exactly 
                            as it appears. Include all numbers, values, 
                            test names, and results. Do not skip anything."""
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error reading image: {str(e)}"

def summarize_report(report_text, language):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a friendly and helpful medical assistant.
                Explain medical reports in very simple language.
                IMPORTANT: You must respond ONLY in {language} language.
                Use bullet points and simple words.
                Always recommend consulting a doctor."""
            },
            {
                "role": "user",
                "content": f"""Please analyze this medical report and provide
the response in {language} language:

1. 📋 What This Report Is About (1-2 sentences)
2. 🔍 Key Findings (bullet points, simple language)
3. ✅ What is Normal (list normal results)
4. ⚠️ What Needs Attention (list abnormal results)
5. 💊 What This Means For You (practical explanation)
6. 🏥 Next Steps (what patient should do)

Use very simple words. Explain medical terms in brackets.
Respond completely in {language}.

Medical Report:
{report_text}"""
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

def analyze_specific_question(report_text, question, language):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful medical assistant.
                Answer questions about medical reports in simple language.
                IMPORTANT: Respond ONLY in {language} language.
                Always recommend consulting a doctor."""
            },
            {
                "role": "user",
                "content": f"""Based on this medical report, answer in {language}:

Question: {question}

Medical Report:
{report_text}

Give a clear simple answer in {language} language."""
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# ── Main App ───────────────────────────────────────────────
st.markdown("### 📤 Upload Your Medical Report")

# ── Upload Type Selector ───────────────────────────────────
upload_type = st.radio(
    "Choose upload type:",
    ["📄 PDF File", "🖼️ Image (JPG/PNG)"],
    horizontal=True
)

report_text = None

if upload_type == "📄 PDF File":
    uploaded_file = st.file_uploader(
        "Upload PDF report",
        type=["pdf"],
    )
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        with st.spinner("📖 Reading PDF..."):
            report_text = extract_text_from_pdf(uploaded_file)

else:
    uploaded_image =

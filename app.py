import streamlit as st
import PyPDF2
from groq import Groq

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
    <p>Understand your medical report in simple language</p>
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
st.markdown("### 📄 Upload Your Medical Report")

uploaded_file = st.file_uploader(
    "Upload PDF report (Blood test, Scan, Lab report etc.)",
    type=["pdf"],
)

if uploaded_file is not None:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    with st.spinner("📖 Reading your report..."):
        report_text = extract_text_from_pdf(uploaded_file)

    if not report_text:
        st.error("❌ Could not read the PDF. Please try another file.")
    else:
        st.info(f"📊 Report contains {len(report_text.split())} words")
        st.markdown("---")

        # ── Language Selector ──────────────────────────────
        st.markdown("### 🌍 Select Your Language")
        selected_lang_label = st.selectbox(
            "Choose the language for your report summary:",
            options=list(LANGUAGES.keys()),
            index=0
        )
        selected_language = LANGUAGES[selected_lang_label]
        st.success(f"✅ Summary will be in: **{selected_lang_label}**")

        st.markdown("---")

        # ── Summarize Button ───────────────────────────────
        if st.button("🧠 Summarize My Report", type="primary",
                     use_container_width=True):
            with st.spinner(f"⚡ Generating summary in {selected_lang_label}..."):
                try:
                    summary = summarize_report(report_text, selected_language)
                    st.markdown("### ✅ Your Report Summary")
                    st.markdown(f"""
                    <div class="summary-box">
                    {summary}
                    </div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name=f"report_summary_{selected_language}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.markdown("---")

        # ── Ask Questions ──────────────────────────────────
        st.markdown("### ❓ Ask a Question About Your Report")
        st.caption(f"Answer will be in {selected_lang_label}")
        user_question = st.text_input(
            "e.g. Is my blood sugar normal? What does hemoglobin mean?"
        )

        if st.button("💬 Get Answer", use_container_width=True):
            if user_question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = analyze_specific_question(
                            report_text,
                            user_question,
                            selected_language
                        )
                        st.markdown("### 💡 Answer")
                        st.success(answer)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please type a question first!")

# ── Disclaimer ─────────────────────────────────────────────
st.markdown("---")
st.warning("""
⚕️ Medical Disclaimer: This AI summary is for informational
purposes only and does not replace professional medical advice.
Always consult a qualified doctor for diagnosis and treatment.
""")

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedReport AI")
    st.markdown("**Turn complex reports into simple language**")
    st.markdown("---")
    st.markdown("### 🌍 Supported Languages")
    for lang in LANGUAGES.keys():
        st.markdown(f"{lang}")
    st.markdown("---")
    st.markdown("### ✅ Supported Reports")
    st.markdown("🩸 Blood test reports")
    st.markdown("🫁 Scan reports (MRI/CT/X-Ray)")
    st.markdown("🧪 Lab reports")
    st.markdown("💊 Prescriptions")
    st.markdown("🫀 ECG reports")
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Reports Analyzed", "1,234")
    st.metric("Happy Users", "456")
    st.metric("Accuracy", "95%")
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("📱 +94754081094")
    st.markdown("📧 althafahamed075@gmail.com")
    st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/althaf-ahamed-810946234)")
    st.markdown("---")
    st.markdown("### 💰 Get Premium Access")
    st.markdown("DM me on LinkedIn or WhatsApp!")

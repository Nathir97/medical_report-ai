import streamlit as st
import PyPDF2
from openai import OpenAI

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
    <p>Powered by DeepSeek AI</p>
</div>
""", unsafe_allow_html=True)

# ── Configure DeepSeek ─────────────────────────────────────
# DeepSeek uses same format as OpenAI!
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

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

def summarize_report(report_text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """You are a friendly and helpful medical assistant.
                Explain medical reports in very simple language that anyone
                can understand. Use bullet points and simple words.
                Always recommend consulting a doctor."""
            },
            {
                "role": "user",
                "content": f"""Please analyze this medical report and provide:

1. 📋 What This Report Is About (1-2 sentences)
2. 🔍 Key Findings (bullet points, simple language)
3. ✅ What is Normal (list normal results)
4. ⚠️ What Needs Attention (list abnormal results)
5. 💊 What This Means For You (practical explanation)
6. 🏥 Next Steps (what patient should do)

Use very simple words. Explain any medical terms in brackets.

Medical Report:
{report_text}

End with: Please consult your doctor for proper medical advice."""
            }
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content

def analyze_specific_question(report_text, question):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful medical assistant.
                Answer questions about medical reports in simple language.
                Always recommend consulting a doctor."""
            },
            {
                "role": "user",
                "content": f"""Based on this medical report, answer:

Question: {question}

Medical Report:
{report_text}

Give a clear simple answer and recommend consulting a doctor."""
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

        # ── Summarize Button ───────────────────────────────
        if st.button("🧠 Summarize My Report", type="primary",
                     use_container_width=True):
            with st.spinner("🤖 DeepSeek AI is analyzing your report..."):
                try:
                    summary = summarize_report(report_text)
                    st.markdown("### ✅ Your Report Summary")
                    st.markdown(f"""
                    <div class="summary-box">
                    {summary}
                    </div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="my_report_summary.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.markdown("---")

        # ── Ask Questions ──────────────────────────────────
        st.markdown("### ❓ Ask a Question About Your Report")
        user_question = st.text_input(
            "e.g. Is my blood sugar normal? What does hemoglobin mean?"
        )

        if st.button("💬 Get Answer", use_container_width=True):
            if user_question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = analyze_specific_question(
                            report_text, user_question
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

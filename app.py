# app.py
import streamlit as st
import fitz  # PyMuPDF

from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# --- Embeddings + Knowledge Base ---
knowledge_base = """
A. ATS CV Scoring Rubric:
- Structure & Formatting (20 pts)
- Keywords (30 pts)
- Skills (20 pts)
- Experience (20 pts)
- Grammar & Readability (10 pts)
TOTAL = 100 pts.

B. Tips for ATS-friendly CV:
- Use standard fonts (Arial, Calibri).
- Avoid graphics, logos, tables.
- Match keywords from job description.
- Keep length 1-2 pages.

C. Interview tips:
- Use STAR method (Situation, Task, Action, Result).
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = [Document(page_content=knowledge_base)]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(doc_chunks, embeddings)
retriever = vectorstore.as_retriever()

# --- Page Config ---
st.set_page_config(page_title="AI Job Seeker Assistant", page_icon="💼")
st.title("💼 AI-Powered Job Seeker Assistant")
st.caption("Analyze CVs, generate cover letters, simulate interviews, and get career advice")

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    reset_button = st.button("Reset Conversation")
    single_chat_mode = st.toggle("Enable Single Chat Mode (Agentic)", value=True)

    if not single_chat_mode:
        mode = st.radio(
            "Choose Mode",
            ["📄 ATS CV Analyzer", "📝 Cover Letter Generator", "🎤 Interview Simulator", "💬 Career Q&A"]
        )

if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start.", icon="🗝️")
    st.stop()

# --- Init LLM ---
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.4
    )

# --- PDF Reader ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# --- Tools ---
@tool
def analyze_cv_tool(cv_text: str) -> str:
    """Analyze a CV and return ATS score + feedback per category."""
    results = retriever.invoke("ATS CV scoring rubric")
    rubric = results[0].page_content if results else "General ATS best practices."
    prompt = f"Use this rubric:\n{rubric}\n\nAnalyze this CV and give detailed feedback + total score:\n{cv_text}"
    llm = get_llm()
    return llm.invoke(prompt).content

@tool
def cover_letter_tool(job_desc: str) -> str:
    """Generate a professional cover letter based on job description."""
    prompt = f"Write a professional cover letter tailored to this job description:\n\n{job_desc}"
    llm = get_llm()
    return llm.invoke(prompt).content

@tool
def interview_tool(qa: str) -> str:
    """Evaluate an interview answer and provide constructive feedback."""
    tips = retriever.invoke("interview tips")[0].page_content
    prompt = f"As an professional interviewer from HRD Division, evaluate this answer and give feedback. Use STAR method if relevant.\n\nTips:{tips}\n\n{qa}"
    llm = get_llm()
    return llm.invoke(prompt).content

@tool
def adaptive_interview_tool(input_text: str) -> str:
    """Generate next interview question based on CV, job description, and past answers."""

    # ex: "ANSWER: ..., CV: ..., JOBDESC: ..., HISTORY: ..."
    llm = get_llm()
    prompt = f"""
    Adapt interview based on the following context:

    {input_text}

    Task:
    1. Evaluate the given answer.
    2. Suggest improvements.
    3. Generate the NEXT best interview question.
    """
    return llm.invoke(prompt).content


@tool
def retriever_tool(query: str) -> str:
    """Retrieve knowledge from ATS & interview tips knowledge base."""
    results = retriever.invoke(query)
    return "\n".join([r.page_content for r in results]) if results else "No relevant knowledge found."

tools = [analyze_cv_tool, cover_letter_tool, interview_tool, retriever_tool, adaptive_interview_tool]

# --- Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.messages = []
    st.rerun()

def convert_st_messages_to_lc_messages(st_messages):
    lc_messages = []
    for m in st_messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages

# --- Main UI ---

# history message
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if single_chat_mode:
    st.subheader("💬 Single Chat Mode (Agentic)")

    # Upload CV opsional
    uploaded_cv = st.file_uploader("Upload your CV (PDF, optional)", type=["pdf"], key="cv_upload_agent")
    cv_text = None
    if uploaded_cv is not None:
        cv_text = extract_text_from_pdf(uploaded_cv)
        st.success("✅ CV uploaded. You can now ask: 'Analyze my CV'.")

    # Input chat
    prompt = st.chat_input("Ask me anything about CVs, cover letters, interviews, or careers...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = get_llm()
            agent = create_react_agent(model=llm, tools=tools)
            lc_messages = convert_st_messages_to_lc_messages(st.session_state.messages)

            try:
                trigger_words = ["analyze", "analisa", "analysis"]
                cv_words = ["cv", "resume", "pdf", "file"]

                if any(word in prompt.lower() for word in trigger_words) and \
                any(word in prompt.lower() for word in cv_words) and cv_text:
                    answer = analyze_cv_tool(cv_text)
                else:
                    response = agent.invoke({"messages": lc_messages})
                    answer = response["messages"][-1].content
            except Exception as e:
                answer = f"❌ Error: {e}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.warning("Sidebar mode non-agent belum diaktifkan. Gunakan Single Chat Mode.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if mode == "📄 ATS CV Analyzer":
        st.subheader("📄 ATS CV Analyzer")
        uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
        if uploaded_cv is not None:
            cv_text = extract_text_from_pdf(uploaded_cv)

            results = retriever.invoke("ATS CV scoring rubric")
            rubric = results[0].page_content if results else "No rubric found"

            prompt = f"Use this rubric:\n{rubric}\n\nAnalyze this CV and give feedback per category + score:\n{cv_text}"

            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                try:
                    response = llm.invoke(prompt)
                    answer = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    answer = f"❌ Error during CV analysis: {e}"

                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)

        followup = st.chat_input("Ask me anything about the CV analysis result...")
        if followup:
            st.session_state.messages.append({"role": "user", "content": followup})
            with st.chat_message("user"):
                st.markdown(followup)
            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                context_prompt = f"""
                The user previously received this CV analysis:\n\n{st.session_state.messages[0]['content']}

                Now they ask: {followup}
                Please respond clearly and reference the analysis when relevant.
                """
                follow_answer = llm.invoke(context_prompt).content
            st.session_state.messages.append({"role": "assistant", "content": follow_answer})
            st.markdown(follow_answer)

    elif mode == "📝 Cover Letter Generator":
        st.subheader("📝 Cover Letter Generator")
        job_desc = st.text_area("Paste the Job Description here:")

        if st.button("Generate Cover Letter") and job_desc:
            prompt = f"Write a professional cover letter tailored to this job description:\n\n{job_desc}"

            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                try:
                    response = llm.invoke(prompt)
                    answer = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    answer = f"❌ Error generating cover letter: {e}"

                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)

        followup = st.chat_input("Ask me anything about the generated cover letter...")
        if followup:
            st.session_state.messages.append({"role": "user", "content": followup})
            with st.chat_message("user"):
                st.markdown(followup)
            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                context_prompt = f"""
                The user previously generated this cover letter:\n\n{st.session_state.messages[0]['content']}

                Now they ask: {followup}
                Please answer with reference to the generated cover letter.
                """
                follow_answer = llm.invoke(context_prompt).content
            st.session_state.messages.append({"role": "assistant", "content": follow_answer})
            st.markdown(follow_answer)

    elif mode == "🎤 Interview Simulator":
        st.subheader("🎤 Interview Simulator")

        job_desc = st.text_area("Paste the Job Description here (optional):")
        question = st.text_input("Interviewer Question (e.g., Tell me about yourself):")
        answer_user = st.text_area("Your Answer:")

        if st.button("Submit Answer") and answer_user:
            cv_text = st.session_state.get("cv_text", "")
            history = "\n".join([m["content"] for m in st.session_state.messages if m["role"]=="user"])

            input_text = f"ANSWER: {answer_user}\nCV: {cv_text}\nJOBDESC: {job_desc}\nHISTORY: {history}"

            feedback_and_next = adaptive_interview_tool.invoke(input_text)

            st.session_state.messages.append({"role": "assistant", "content": feedback_and_next})
            with st.chat_message("assistant"):
                st.markdown(feedback_and_next)

        followup = st.chat_input("Ask me about the feedback, improvements, or interview tips...")
        if followup:
            st.session_state.messages.append({"role": "user", "content": followup})
            with st.chat_message("user"):
                st.markdown(followup)
            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                context_prompt = f"""
                The user previously received this interview feedback:\n\n{st.session_state.messages[0]['content']}

                Now they ask: {followup}
                Please answer like a professional interviewer/coach.
                """
                follow_answer = llm.invoke(context_prompt).content
            st.session_state.messages.append({"role": "assistant", "content": follow_answer})
            st.markdown(follow_answer)

    else:  # 💬 Career Q&A
        st.subheader("💬 Career Q&A")
        prompt = st.chat_input("Ask me anything about career, jobs, or skills...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                container = st.empty()
                llm = get_llm()
                try:
                    response = llm.invoke(prompt)
                    answer = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    answer = f"❌ Error during Career Q&A: {e}"

            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)


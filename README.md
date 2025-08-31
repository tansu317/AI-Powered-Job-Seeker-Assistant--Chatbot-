💼 **AI-Powered Job Seeker Assistant**

📌 Project Overview
AI Job Seeker Assistant adalah aplikasi chatbot interaktif berbasis Streamlit yang dirancang untuk membantu pencari kerja mempersiapkan diri lebih baik.
Aplikasi ini memanfaatkan LLM (Gemini), RAG (Retrieval Augmented Generation), dan ReAct Agent untuk:

1. Menganalisis CV secara otomatis menggunakan rubric ATS.
2. Membuat cover letter yang relevan dengan job description.
3. Menyimulasikan interview dan memberi feedback berbasis STAR method.
4. Menjawab pertanyaan seputar karier & skill dengan konteks dari knowledge base.


✨ Fitur Utama
✅ ATS CV Analyzer
✅ Cover Letter Generator
✅ Interview Simulator
✅ Career Q&A
✅ Agentic Mode (Single Chat) dengan ReAct Agent


🧠 Model AI yang Digunakan
1. Google Gemini 1.5 Flash → LLM utama
2. HuggingFace MiniLM Embeddings → vektor embedding untuk RAG


flowchart TD

    A[User Input] -->|CV / JD / Pertanyaan| B[Streamlit UI]
    B --> C{ReAct Agent}
    
    C -->|CV Analysis| D[CV Analyzer Tool]
    C -->|Cover Letter| E[Cover Letter Generator Tool]
    C -->|Interview| F[Interview Feedback Tool]
    C -->|RAG Query| G[Retriever Tool]

    G --> H[FAISS Vectorstore]
    H --> G

    C --> I[LLM: Gemini 1.5 Flash]
    D --> I
    E --> I
    F --> I
    G --> I

    I --> J[Chat Response]
    J --> B

Penjelasan Diagram:
- User Input → pengguna bisa upload CV, masukkan job description, atau bertanya di chat.
- Streamlit UI → interface chat & sidebar.
- ReAct Agent → memutuskan tool mana yang dipanggil.
- Retriever Tool (RAG) → mencari data relevan di knowledge base.
- LLM (Gemini) → melakukan reasoning + menghasilkan jawaban.
- Chat Response → hasil dikirim kembali ke pengguna.

RAG + ReAct Agent:
1. Menggunakan RAG retriever → memberi context relevan ke LLM.
2. ReAct agent → memastikan jawaban grounded ke tool & data.
3. Menyertakan saran dari knowledge base → meminimalisir halusinasi pada hasil analyzer CV ATS.

ReAct Agent Reasoning Flow
sequenceDiagram

    participant U as 👤 User
    participant A as 🤖 ReAct Agent
    participant T as 🛠️ Tool
    participant L as 🧠 LLM (Gemini)
    
    U->>A: "Analisa CV saya"
    A->>L: Thought: "Butuh data CV, pakai CV Analyzer"
    A->>T: Action: CV Analyzer Tool(input=CV)
    T-->>A: Observation: "Skor ATS: 78%, Saran: perbaiki skill section"
    A->>L: Thought: "Tambahkan saran ke jawaban final"
    A-->>U: Final Answer: "Skor ATS kamu 78%, perbaiki skill section agar lebih ATS-friendly."

🖥️ Teknologi & Library yang Digunakan
1. Streamlit
2. LangChain + LangGraph
3. FAISS
4. HuggingFace Sentence Transformers
5. PyMuPDF
6. Google Gemini API

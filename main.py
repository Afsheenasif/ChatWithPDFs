import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page settings
st.set_page_config(
    page_title="PilotPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page title
st.markdown("""
    <h1>📚 DocQuery Pilot</h1>
    <h5>AI-Powered Document Query Assistant</h5>
""", unsafe_allow_html=True)

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF text extraction
def get_pdf_txt(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# Chunk text for embeddings
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create FAISS vector store
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

# Build QA chain
def get_conversational_chain():
    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context.
        If the answer is not found in the context, respond with:
        \"Answer is not available in the context.\"

        Context: \n{context}\n
        Question: \n{question}\n
        Answer:
        """,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User input handling
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    chain = get_conversational_chain()
    with st.spinner("🤖 Bot is typing..."):
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    st.session_state.chat_history.append({"role": "user", "text": question})
    st.session_state.chat_history.append({"role": "assistant", "text": response["output_text"]})


def main():
    # Chat form
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([9, 1])
        with cols[0]:
            question = st.text_input("", key="user_question", placeholder="Enter Your Question Here...",
                                     label_visibility="collapsed")
        with cols[1]:
            submitted = st.form_submit_button("➤")

    if submitted and question:
        user_input(question)

    # Display chat
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)

    # Group messages into Q&A pairs
    pairs = []
    chat = st.session_state.chat_history
    i = 0
    while i < len(chat) - 1:
        if chat[i]["role"] == "user" and chat[i + 1]["role"] == "assistant":
            pairs.append((chat[i], chat[i + 1]))
            i += 2
        else:
            i += 1  # skip malformed entries

    # Reverse to show latest first
    for user_msg, bot_msg in reversed(pairs):
        # User message (left 3 cols)
        user_col, spacer1, spacer2, _ = st.columns([3, 0.2, 0.2, 3])
        with user_col:
            st.image("user_icon.jpeg", width=36)
            st.markdown(f'<div class="user-message">{user_msg["text"]}</div>', unsafe_allow_html=True)

        # Bot message (right 3 cols)
        _, spacer1, spacer2, bot_col = st.columns([3, 0.2, 0.2, 3])
        with bot_col:
            st.image("bot_icon.jpeg", width=36)
            st.markdown(f'<div class="bot-message">{bot_msg["text"]}</div>', unsafe_allow_html=True)

    # Sidebar for file upload and control
    with st.sidebar:
        st.markdown("## 📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("📄 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_txt(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")




if __name__=="__main__":
    main()

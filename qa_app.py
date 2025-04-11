import os
import re
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import gradio as gr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# 🔐 Load Hugging Face API token from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 🤖 Set up LLM and embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

# 🧼 Utility to sanitize filenames
def clean_name(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem)

# 📄 PDF processor: load or create vectorstore index
def process_pdf(file):
    pdf_name = clean_name(file.name)
    index_path = os.path.join("data", "vectorstores", pdf_name)

    if os.path.exists(index_path):
        print(f"📦 Loading existing index: {pdf_name}")
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print(f"🆕 Creating new index: {pdf_name}")
        loader = PyPDFLoader(file.name)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embedding_model)
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        print(f"✅ Saved new index at: {index_path}")

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )
    return qa_chain

# 🎯 Gradio action handlers
def handle_upload(file):
    qa_chain = process_pdf(file)
    return qa_chain, "✅ PDF processed! You can now ask questions."

def ask_question(qa_chain, question):
    if qa_chain is None:
        return "⚠️ Please upload and process a PDF first."
    return qa_chain.run(question)

# 🎛️ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Upload a PDF and Ask Questions")

    qa_chain_state = gr.State()

    with gr.Row():
        file = gr.File(label="Upload PDF", file_types=[".pdf"])
        file_button = gr.Button("Process PDF")

    with gr.Row():
        question = gr.Textbox(label="Ask a question about the document")
        answer = gr.Textbox(label="Answer", lines=4)

    file_button.click(fn=handle_upload, inputs=file, outputs=[qa_chain_state, answer])
    question.submit(fn=ask_question, inputs=[qa_chain_state, question], outputs=answer)

if __name__ == "__main__":
    demo.launch()

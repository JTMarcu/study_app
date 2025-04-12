import os
import re
import csv
import json
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import gradio as gr

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === Load .env securely ===
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Model & Embedding setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512
)

# === Prompt templates ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}

Respond with a concise and factual answer based only on the context. 
If the answer is not in the context, say: "I couldn't find that in the document."
"""
)

flashcard_prompt = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are an expert teacher. Based on the following study material, generate 2-3 helpful flashcards.
Return a JSON array of objects like:
[
  {"term": "...", "definition": "...", "category": "..."},
  ...
]

Text:
{chunk}
"""
)

flashcard_chain = flashcard_prompt | llm

# === Utilities ===
def clean_name(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem)

def load_or_create_vectorstore(file):
    pdf_name = clean_name(file.name)
    path = f"data/vectorstores/{pdf_name}"
    if os.path.exists(path):
        vs = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    else:
        docs = PyPDFLoader(file.name).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
        vs = FAISS.from_documents(chunks, embedding_model)
        vs.save_local(path)
    return vs, pdf_name

def build_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": qa_prompt}
    )

def generate_flashcards_from_vectorstore(vectorstore, pdf_name, max_chunks=25):
    chunks = [doc.page_content for doc in vectorstore.docstore._dict.values()][:max_chunks]
    flashcards = []

    for i, chunk in enumerate(chunks):
        try:
            response = flashcard_chain.invoke({"chunk": chunk})
            cards = json.loads(response)
            if isinstance(cards, dict):
                flashcards.append(cards)
            elif isinstance(cards, list):
                flashcards.extend(cards)
        except Exception as e:
            print(f"❌ Chunk {i+1} error:", e)

    # Save CSV
    output_csv = Path("data/flashcards") / f"{pdf_name}_flashcards.csv"
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "definition", "category"])
        writer.writeheader()
        for card in flashcards:
            writer.writerow({
                "term": card.get("term", "").strip(),
                "definition": card.get("definition", "").strip(),
                "category": card.get("category", "General").strip()
            })

    return flashcards

# === Gradio Logic ===
def upload_and_index_pdf(file):
    vectorstore, pdf_name = load_or_create_vectorstore(file)
    qa_chain = build_qa_chain(vectorstore)
    return qa_chain, vectorstore, pdf_name, f"✅ {file.name} loaded."

def answer_question(qa_chain, question):
    if not qa_chain:
        return "❗ Please upload a PDF first."
    return qa_chain.run(question)

def show_flashcards(vectorstore, pdf_name):
    flashcards = generate_flashcards_from_vectorstore(vectorstore, pdf_name, max_chunks=25)
    return "\n\n".join([
        f"📘 **{card['term']}**\n{card['definition']} _(Category: {card.get('category', 'General')})_"
        for card in flashcards
    ])

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Study Assistant")

    qa_chain_state = gr.State()
    vectorstore_state = gr.State()
    pdf_name_state = gr.State()

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
        answer = gr.Textbox(label="Answer", lines=4)

    flashcard_btn = gr.Button("Generate Flashcards (25 chunks max)")
    flashcard_display = gr.Textbox(label="Flashcards", lines=20)

    upload_btn.click(upload_and_index_pdf, inputs=pdf_file, outputs=[qa_chain_state, vectorstore_state, pdf_name_state, status])
    question.submit(answer_question, inputs=[qa_chain_state, question], outputs=answer)
    flashcard_btn.click(show_flashcards, inputs=[vectorstore_state, pdf_name_state], outputs=flashcard_display)

if __name__ == "__main__":
    demo.launch()

import os
import re
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

# === Load .env and setup token ===
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Embedding & LLM setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512
)

# === Prompt Templates ===
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

key_term_prompt = PromptTemplate(
    input_variables=["chunk"],
    template="""
Extract up to 5 important technical terms from the following text. 
Return them as a JSON list of strings (e.g., ["CPU", "RAM", ...]).

Text:
{chunk}
"""
)

definition_prompt = PromptTemplate(
    input_variables=["terms"],
    template="""
Define the following technical terms in a helpful, concise way. 
Return a JSON array of {"term": ..., "definition": ...} objects.

Terms:
{terms}
"""
)

term_extractor = key_term_prompt | llm
definition_chain = definition_prompt | llm

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

# === Flashcard generation (2-step) ===
def extract_and_define_terms(vectorstore, pdf_name, max_chunks=25):
    chunks = [doc.page_content for doc in vectorstore.docstore._dict.values()][:max_chunks]
    term_set = set()

    for chunk in chunks:
        try:
            response = term_extractor.invoke({"chunk": chunk})
            terms = json.loads(response)
            if isinstance(terms, list):
                term_set.update([t.strip() for t in terms])
        except Exception as e:
            print("❌ Term extraction error:", e)

    print(f"✅ Extracted {len(term_set)} unique terms.")

    terms_list = list(term_set)
    flashcards = []

    batch_size = 10
    for i in range(0, len(terms_list), batch_size):
        batch = terms_list[i:i+batch_size]
        try:
            response = definition_chain.invoke({"terms": json.dumps(batch)})
            defs = json.loads(response)
            flashcards.extend(defs)
        except Exception as e:
            print("❌ Definition error:", e)

    # Save as JSON for now
    output_path = Path("data/flashcards") / f"{pdf_name}_flashcards.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flashcards, f, indent=2)

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

def generate_and_show_flashcards(vectorstore, pdf_name):
    flashcards = extract_and_define_terms(vectorstore, pdf_name, max_chunks=25)
    return "\n\n".join([
        f"📘 **{card['term']}**\n{card['definition']}"
        for card in flashcards
    ])

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Study Assistant (Key Term Flashcards)")

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

    flashcard_btn = gr.Button("Generate Flashcards (Key Terms + Definitions)")
    flashcard_display = gr.Textbox(label="Flashcards", lines=20)

    upload_btn.click(upload_and_index_pdf, inputs=pdf_file, outputs=[qa_chain_state, vectorstore_state, pdf_name_state, status])
    question.submit(answer_question, inputs=[qa_chain_state, question], outputs=answer)
    flashcard_btn.click(generate_and_show_flashcards, inputs=[vectorstore_state, pdf_name_state], outputs=flashcard_display)

if __name__ == "__main__":
    demo.launch()

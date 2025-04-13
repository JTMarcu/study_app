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

# === Load Hugging Face API token from .env ===
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
hf_token = config.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN not found in .env!")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# === Embedding model and LLM setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# === Prompt templates ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}

Respond with a concise and accurate answer based only on the context above.
If the answer is not found in the context, say: "I couldn't find that in the document."
"""
)

term_prompt = PromptTemplate(
    input_variables=["chunk"],
    template="""
Extract up to 3 important study terms from the text below. Return them as a JSON array of strings.

Text:
{chunk}
"""
)

definition_prompt = PromptTemplate(
    input_variables=["terms"],
    template="""
Define the following study terms clearly and concisely.
Return a JSON array of {{"term": "...", "definition": "..."}} objects.

Terms:
{terms}
"""
)

term_chain = term_prompt | llm
definition_chain = definition_prompt | llm

# === Utilities ===
def extract_json_array(text):
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def clean_name(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem)

def load_or_create_vectorstore(file):
    name = clean_name(file.name)
    index_path = f"data/vectorstores/{name}"

    if os.path.exists(index_path):
        vs = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        docs = PyPDFLoader(file.name).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
        vs = FAISS.from_documents(chunks, embedding_model)
        vs.save_local(index_path)

    return vs, name

def build_qa_chain(vs):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(),
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": qa_prompt}
    )

def extract_and_define_terms(vs, name, max_chunks=10):
    chunks = [doc.page_content for doc in vs.docstore._dict.values()][:max_chunks]
    terms = set()

    for i, chunk in enumerate(chunks):
        try:
            result = term_chain.invoke({"chunk": chunk}).strip()
            print(f"🧠 Chunk {i+1}: {result}")
            parsed = extract_json_array(result)
            if parsed:
                terms.update([t.strip() for t in parsed if isinstance(t, str)])
        except Exception as e:
            print(f"❌ Term extraction error on chunk {i+1}: {e}")

    flashcards = []
    term_list = list(terms)

    for i in range(0, len(term_list), 10):
        batch = term_list[i:i + 10]
        try:
            result = definition_chain.invoke({"terms": json.dumps(batch)}).strip()
            print(f"📘 Definitions batch {i//10+1}: {result}")
            definitions = extract_json_array(result)
            if definitions:
                flashcards.extend(definitions)
        except Exception as e:
            print(f"❌ Definition batch {i//10+1} error: {e}")

    output_path = Path("data/flashcards") / f"{name}_flashcards.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flashcards, f, indent=2)

    return flashcards

# === Gradio Actions ===
def upload_and_process(file):
    vs, name = load_or_create_vectorstore(file)
    chain = build_qa_chain(vs)
    return chain, vs, name, f"✅ {file.name} processed."

def ask(chain, q):
    if not chain:
        return "⚠️ Please upload a PDF first."
    return chain.run(q)

def create_flashcards(vs, name):
    cards = extract_and_define_terms(vs, name)
    return "\n\n".join([
        f"📘 **{c['term']}**\n{c['definition']}"
        for c in cards if "term" in c and "definition" in c
    ])

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 📚 PDF Study Assistant")

    qa_chain_state = gr.State()
    vectorstore_state = gr.State()
    name_state = gr.State()

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
        answer = gr.Textbox(label="Answer", lines=4)

    flashcard_btn = gr.Button("Generate Flashcards")
    flashcards = gr.Textbox(label="Flashcards", lines=20)

    process_btn.click(upload_and_process, inputs=pdf, outputs=[qa_chain_state, vectorstore_state, name_state, status])
    question.submit(ask, inputs=[qa_chain_state, question], outputs=answer)
    flashcard_btn.click(create_flashcards, inputs=[vectorstore_state, name_state], outputs=flashcards)

if __name__ == "__main__":
    demo.launch()

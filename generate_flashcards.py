import os
import re
import csv
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from tkinter import Tk, filedialog

# 🔐 Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
config = dotenv_values(dotenv_path=env_path)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 🤖 LLM and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    token=hf_token,
    temperature=0.5,
    max_new_tokens=512,
    task="text-generation"  # ✅ Important!
)

# 🧠 Prompt template
flashcard_prompt = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are an expert teacher. Based on the following study material, generate 2-3 helpful flashcards.
Each flashcard should be a JSON object with "term", "definition", and an optional "category".

Text:
{chunk}

Return JSON only:
"""
)

chain = flashcard_prompt | llm

# 🧼 Sanitize filename
def clean_name(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem)

# 📥 Main logic
def generate_flashcards_from_pdf(pdf_path: Path):
    cleaned_name = clean_name(pdf_path.name)
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"📄 Processing {len(chunks)} chunks from {pdf_path.name}")

    flashcards = []
    for i, chunk in enumerate(chunks):
        print(f"⚙️ Chunk {i+1}/{len(chunks)}...")
        try:
            response = chain.invoke({"chunk": chunk.page_content})
            cards = eval(response)  # Trusted model, otherwise use json.loads
            if isinstance(cards, dict):
                flashcards.append(cards)
            elif isinstance(cards, list):
                flashcards.extend(cards)
        except Exception as e:
            print(f"❌ Error in chunk {i+1}: {e}")

    output_path = Path("data/flashcards") / f"{cleaned_name}_flashcards.csv"
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["term", "definition", "category"])
        writer.writeheader()
        for card in flashcards:
            writer.writerow({
                "term": card.get("term", "").strip(),
                "definition": card.get("definition", "").strip(),
                "category": card.get("category", "General").strip()
            })

    print(f"✅ Flashcards saved to: {output_path}")
    return output_path

# 🖼️ File picker launcher
def pick_pdf():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.askopenfilename(
        title="Select a PDF to generate flashcards",
        filetypes=[("PDF files", "*.pdf")]
    )
    root.destroy()
    return Path(file_path) if file_path else None

# ▶️ Run
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate flashcards from a PDF.")
    parser.add_argument("pdf", nargs="?", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else pick_pdf()

    if not pdf_path or not pdf_path.exists():
        print("❌ No valid PDF selected.")
    else:
        generate_flashcards_from_pdf(pdf_path)

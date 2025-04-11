## 🧾 PDF Q&A App with LangChain and Gradio

This project allows you to upload any PDF document and ask questions about its contents. Using a combination of **LangChain**, **Hugging Face LLMs**, and **FAISS vectorstores**, the app accurately retrieves answers grounded in the actual content of the document — no hallucinations!

---

### 🚀 Features

- 📄 **Upload a PDF** and convert it into a searchable knowledge base
- 🧠 **Ask questions** and get accurate, grounded answers from the document
- 💾 **Saves vectorstores** to avoid reprocessing the same PDFs
- 🔍 Uses a custom prompt template to reduce hallucination
- 🌐 Simple and clean **Gradio web interface**

---

### 🛠️ Tech Stack

- **LangChain** — for document loading, vector search, and retrieval-based Q&A
- **Hugging Face** — to connect to `mistralai/Mistral-7B-Instruct-v0.1` (or `flan-t5-base`)
- **FAISS** — fast and efficient similarity search for embeddings
- **Gradio** — no-friction UI for uploading, asking, and answering
- **Python-dotenv** — securely loads your Hugging Face API token

---

### 📂 Folder Structure

```
study_app/
├── qa_app.py              ← Main application script
├── .env                   ← Your Hugging Face API token (not tracked)
├── data/
│   └── vectorstores/      ← Stores FAISS indexes by PDF name
```

---

### 🔐 Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root:
   ```
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   ```

3. Run the app:
   ```bash
   python qa_app.py
   ```

---

### ✍️ Prompt Logic

This app uses the following prompt to keep responses faithful to the document:

```
You are a helpful assistant. Use only the provided context to answer the question as accurately and clearly as possible.

If the answer cannot be found in the context, say:
"I couldn't find that in the document."
```

---

### 📌 Notes

- If a vectorstore already exists for a PDF, it will be loaded automatically.
- If the document is new, it will be processed and saved for future use.
- This app does **not** hallucinate — it uses only the PDF content for answers.

---

### 🧠 Coming Soon (Optional Ideas)

- Flashcard generator (`flashcards.py`)
- Mock exam builder
- Drop-down to select previous documents
- Export to Anki deck

---

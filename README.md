# 🧠 PDF Study Assistant

**An AI-powered tool to help you master any PDF material through efficient, accurate, and interactive study methods.**
This application leverages **LangChain**, **FAISS Vectorstores**, and **Hugging Face LLMs** to extract, understand, and study critical concepts from any PDF you provide—be it textbooks, research papers, manuals, or study guidesIt features an integrated interactive flashcard system, adapted from [Jonathan Marcu's Flashcard App](https://jtmarcu.github.io/projects/flashcards.html), now utilizing JSON for data storage

---

## 🎯 Project Goals

- ✅ **Precision** Generate content and answers strictly sourced from your provided PDF to ensure reliability and accurac.
- ✅ **Automation** Automatically create flashcards, define key terms, and support interactive Q&A sessions from extensive document.
- ✅ **Versatility** Accommodate a wide range of PDF materials, ensuring broad use for various study or review purpose.
- ✅ **Interactive Learning** Provide an intuitive Gradio interface combined with dynamic flashcards for an effective learning environmen.

---

## 🚀 Key Features

- 📚 **PDF Upload & Processing*: Convert any PDF into a searchable knowledge base using embeddings and vectorstoes.
- 🔎 **Interactive Q&**: Ask focused questions and receive accurate, context-specific answers based on your docuent.
- 🗂️ **Flashcard Generatin**: Automatically identify and define key terms directly from your PDF, simplifying memorization and rview.
- 💾 **Persistent Storge**: Save processed documents and flashcards for faster future ccess.
- 📖 **Intuitive Interace**: User-friendly Gradio app for effortless inteaction.

---

## 🧑‍💻 Technology Stack

- **LanChain**: Document processing, embedding creation, and retrieval-ased Q&A.
- **Huggin Face**: Access powerful language models (e.g., Mistral-7B-Instruct) for generating flashcard and Q&A.
- *FAISS**: Fast and efficient similarity search for effective documentindexing.
- **radio**: Intuitive, web-based user nterface.
- *Flask**: Backend framework for serving the interactive flashcad system.
- **JavaScript, HTM, CSS**: Frontend technologies for the flashcard nterface.
- **Python-otenv**: Secure handling of AI tokens.

---

## 📂 Project Structure

```
study_app/
├── app.py                    ← Main Gradio application (PDF upload, Q&A, flashcards)
├── flashcards/
│   ├── templates/
│   │   └── index.html        ← Flashcard interface template
│   ├── static/
│   │   ├── styles.css        ← Flashcard styles
│   │   └── script.js         ← Flashcard logic
│   └── flashcards.json       ← Generated flashcards in JSON format
├── .env                      ← Hugging Face API token (secure & not tracked)
├── data/
│   └── vectorstores/         ← FAISS indexes for PDFs
├── requirements.txt          ← Python dependencies
└── README.md                 ← Project documentation and usage guidelines
```

---

## 🔧 Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone your-repo-link
   cd study_app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your `.env` File**
   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   ```

4. **Launch the App**
   ```bash
   python app.py
   ```

---

## 📚 How to Use the App

### Upoad a PDF
- Drag-and-drop or browse your DF dcument.
- The PDF is automatically processed and indexed for efficiet retrieval.

### AskQuestions
- Enter questions about yourPDF ontent.
- Receive accurate and concise answers directly derived from yur document.

### Generate lashcards
- Automatically create clear definitions of critical terms nd cncepts.
- Review these flashcards interactively through the integrated flashcad interface.

---

## 🎓 Best Practices

- **High-Qulity PDFs**: Clearly formatted documents provide thebest results.
- **Prompt Veification**: Always verify generated answers and flashcardsfor accuracy.
- **Cost fficiency**: Limit the number of terms or document sections processed when testing to manage inference cost effectively.

---

## 🗒️ Future Developmen Plans

- [ ] Automated Moc Exam Cration
- [ ] Export Flahcards t Anki
- [ ] Enhanced UI Features (e.g., multiple-document handling, progrss indictors)
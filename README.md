# ApexcifyTechnologys_Chatbot-for-FAQs

# FAST-NUCES FAQ Chatbot

An NLP-powered FAQ chatbot for FAST-NUCES university queries, built with **Flask**, **NLTK**, and **scikit-learn**. Users can type natural language questions and get the most relevant answer using **TF-IDF cosine similarity** matching.

---

## Features

- NLP preprocessing pipeline (tokenization, stopword removal, stemming)
- TF-IDF vectorization using scikit-learn
- Cosine similarity for FAQ matching
- Confidence scoring (High / Medium / Low)
- Clean, minimal chat UI (HTML + CSS + JavaScript)
- Flask REST API backend

---

## Tech Stack

| Layer    | Technology                        |
|----------|-----------------------------------|
| Backend  | Python, Flask                     |
| NLP      | NLTK (tokenizer, stopwords, Porter Stemmer) |
| ML       | scikit-learn (TfidfVectorizer, cosine_similarity) |
| Frontend | HTML, CSS, JavaScript (Fetch API) |

---

## Project Structure
```
fast-faq-chatbot/
├── app.py              # Flask backend — NLP pipeline + API
└── templates/
    └── index.html      # Chat UI frontend
```

>`index.html` **must** be inside a folder named `templates/` — this is required by Flask.

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/AbeelaKhan/ApexcifyTechnologys_Chatbot-for-FAQs
cd ApexcifyTechnologys_Chatbot-for-FAQs
```

### 2. Install dependencies
```bash
pip install flask nltk scikit-learn
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## How It Works

### NLP Pipeline (inside `app.py`)

Each FAQ question + answer goes through this pipeline before matching:
```
Raw Text
   ↓
Tokenization        → word_tokenize()  [NLTK]
   ↓
Lowercase + Clean   → remove punctuation & non-alphabetic tokens
   ↓
Stopword Removal    → remove "the", "is", "what", "are" etc.
   ↓
Stemming            → PorterStemmer  e.g. "admission" → "admiss"
   ↓
TF-IDF Vector       → TfidfVectorizer  [scikit-learn]
   ↓
Cosine Similarity   → match against all FAQ vectors → best answer
```

### API Endpoint

**POST** `/chat`

Request:
```json
{ "message": "What are the tuition fees?" }
```

Response:
```json
{
  "answer": "Tuition fees vary by campus...",
  "matched_question": "How much are the tuition fees?",
  "score": 0.631,
  "confidence": "high"
}
```

### Confidence Levels

| Score      | Confidence |
|------------|------------|
| ≥ 0.25     | 🟢 High     |
| 0.08–0.24  | 🟡 Medium   |
| < 0.08     | 🔴 Low      |

---

## Sample Questions to Try

- What are the tuition fees?
- How do I apply for admission?
- Are scholarships available?
- Which campuses does FAST have?
- Is hostel accommodation available?
- What is the grading system?
- What programs are offered?

---

## Libraries Used

- [Flask](https://flask.palletsprojects.com/) — web framework
- [NLTK](https://www.nltk.org/) — natural language toolkit
- [scikit-learn](https://scikit-learn.org/) — TF-IDF & cosine similarity

---

## Author

**AbeelaKhan** — BS Computer Science, FAST-NUCES Karachi  
AI Internship Task 2 · Apexcify Technologies

from flask import Flask, request, jsonify, render_template
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('stopwords',    quiet=True)

app = Flask(__name__)

faqs = [
    {
        "question": "How much are the tuition fees?",
        "answer": "Tuition fees vary by campus and program. For undergraduate programs, the semester fee ranges from PKR 90,000 to PKR 120,000. Graduate program fees range from PKR 60,000 to PKR 100,000 per semester. Additional charges may apply for labs and other facilities."
    },
    {
        "question": "What are the admission requirements for undergraduate programs?",
        "answer": "Admission requires at least 60% marks in FSc/A-Levels or equivalent. You must pass the NU Entry Test (NET) or have valid SAT/HAT scores. Required documents include matric & intermediate certificates, CNIC/B-Form, passport photos, and domicile certificate."
    },
    {
        "question": "What is the duration of undergraduate programs?",
        "answer": "Undergraduate programs at FAST-NUCES are typically 4 years (8 semesters) in duration. Students must complete all credit hours and meet GPA requirements to graduate."
    },
    {
        "question": "Is hostel accommodation available?",
        "answer": "Yes, hostel accommodation is available at most FAST-NUCES campuses for both male and female students. Hostels are equipped with Wi-Fi, 24/7 security, and dining facilities. Early application is recommended as seats are limited."
    },
    {
        "question": "What scholarships are available at FAST-NUCES?",
        "answer": "FAST-NUCES offers need-based and merit-based scholarships. HEC provides various scholarship programs including need-based scholarships. Students with exceptional academic performance may receive fee waivers. Contact the financial aid office for detailed eligibility criteria."
    },
    {
        "question": "Which campuses does FAST-NUCES have?",
        "answer": "FAST-NUCES has campuses in Karachi, Lahore, Islamabad, Peshawar, Chiniot-Faisalabad, and Hyderabad. Each campus offers a range of undergraduate and graduate programs in computing and business."
    },
    {
        "question": "How can I contact FAST-NUCES admissions office?",
        "answer": "You can contact FAST-NUCES via their official website at nu.edu.pk. Each campus has a dedicated admissions office with contact numbers and email addresses listed on the website. You can also visit in person during working hours."
    },
    {
        "question": "What programs are offered at FAST-NUCES?",
        "answer": "FAST-NUCES offers BS Computer Science, BS Software Engineering, BS Data Science, BS Artificial Intelligence, BS Cyber Security, BBA, MS/PhD programs in CS and related fields, and MBA programs."
    },
    {
        "question": "When does the admission process start?",
        "answer": "FAST-NUCES typically opens admissions twice a year — Spring (January-February) and Fall (June-July). The NU Entry Test (NET) schedule is announced on the official website. Keep checking nu.edu.pk for updated dates."
    },
    {
        "question": "What is the grading system at FAST-NUCES?",
        "answer": "FAST-NUCES uses a 4.0 GPA scale. A grade is 4.0, B is 3.0, C is 2.0, D is 1.0, and F is 0.0. Students must maintain a minimum CGPA of 2.0 to remain in good academic standing."
    },
    {
        "question": "Are there any student societies or clubs?",
        "answer": "Yes! FAST-NUCES has a vibrant student life with societies covering computing (ACM, IEEE), arts, sports, entrepreneurship, and more."
    },
    {
        "question": "What is the fee payment method?",
        "answer": "Fees can be paid via bank challan, online banking, or through the student portal."
    },
    {
        "question": "Is there a library at FAST-NUCES?",
        "answer": "Yes, each campus has a well-equipped library with books, journals, and digital resources."
    },
    {
        "question": "What is the minimum attendance requirement?",
        "answer": "Students must maintain at least 80% attendance in each course."
    },
    {
        "question": "Does FAST-NUCES offer internship opportunities?",
        "answer": "Yes, FAST has a Career Development Center that connects students with internships and jobs."
    }
]

stemmer   = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Full NLP preprocessing pipeline:
    1. Tokenize the text into individual words
    2. Lowercase all tokens
    3. Remove punctuation tokens
    4. Remove English stopwords
    5. Apply Porter Stemming to reduce words to root form
    Returns a single cleaned string (needed by TfidfVectorizer).
    """
    tokens = word_tokenize(text.lower())

    tokens = [t for t in tokens if t not in string.punctuation and t.isalpha()]

    tokens = [t for t in tokens if t not in stop_words]

    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)



faq_documents = [
    preprocess(f["question"] + " " + f["answer"])
    for f in faqs
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_documents)

print("NLP pipeline ready. TF-IDF matrix shape:", tfidf_matrix.shape)


def get_best_match(user_query):
    """
    Preprocesses the user query, converts it to a TF-IDF vector,
    then computes cosine similarity against all FAQ vectors.
    Returns the best matching FAQ and its similarity score.
    """
    cleaned_query = preprocess(user_query)

    if not cleaned_query.strip():
        return None, 0.0

    query_vector = vectorizer.transform([cleaned_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    best_index = similarities.argmax()
    best_score = float(similarities[best_index])

    return faqs[best_index], best_score


@app.route("/")
def index():
    """Serve the HTML frontend."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Accepts JSON: { "message": "user question here" }
    Returns JSON: { "answer": "...", "matched_question": "...", "score": 0.xx, "confidence": "high/medium/low" }
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    best_faq, score = get_best_match(user_message)

    if score >= 0.25:
        confidence = "high"
    elif score >= 0.08:
        confidence = "medium"
    else:
        confidence = "low"

    if score < 0.04 or best_faq is None:
        return jsonify({
            "answer": "I'm sorry, I couldn't find a relevant answer to your question. Try asking about fees, admissions, programs, scholarships, or hostels.",
            "matched_question": None,
            "score": round(score, 4),
            "confidence": "none"
        })

    return jsonify({
        "answer": best_faq["answer"],
        "matched_question": best_faq["question"],
        "score": round(score, 4),
        "confidence": confidence
    })


#  RUN THE APP
if __name__ == "__main__":
    print("Starting FAST-NUCES FAQ Chatbot...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)

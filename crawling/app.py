from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import spacy
import PyPDF2
import pytesseract
import shutil
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = ' '.join(raw_text.split())
                    text += cleaned_text + "\n"

        # If PyPDF2 fails to extract text, use OCR
        if not text.strip():
            text = extract_text_with_ocr(pdf_path)

    except Exception as e:
        return None, str(e)
    return text, None

def extract_text_with_ocr(pdf_path):

    text = ""
    try:
        images = convert_from_path(pdf_path, dpi=300)
        for image in images:
            text += pytesseract.image_to_string(image, lang="eng") + "\n"
    except Exception as e:
        return f"Error during OCR: {e}"
    return text

def analyze_context(content):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(content)
    relevant_roles = []

    for entity in doc.ents:
        if entity.label_ in ["EDUCATION", "SKILLS", "WORK_OF_ART", "ORG", "GPE"]:
            relevant_roles.append(entity.text)

    for chunk in doc.noun_chunks:
        relevant_roles.append(chunk.text)

    return list(set(relevant_roles))  

def score_content(content, keyword_weights):

    score = 0
    max_score = 100
    nlp = spacy.load("en_core_web_sm")
    sentences = sent_tokenize(content)
    missing_scores = {key: weight for key, weight in keyword_weights.items()}
    keyword_frequency = {key: 0 for key in keyword_weights.keys()}

    for sentence in sentences:
        doc = nlp(sentence.lower())
        words = set(token.text for token in doc if token.is_alpha)  # Use unique words only
        for keyword, weight in keyword_weights.items():
            if keyword in words:
                keyword_frequency[keyword] += 1
                if keyword in missing_scores:
                    score += weight
                    del missing_scores[keyword]

    
    if "project" in missing_scores:
        score += 15
        del missing_scores["project"]
        keyword_frequency["project"] = 1

   
    score = min(score, max_score)

    return score, missing_scores, keyword_frequency

def visualize_evaluation(missing_scores, evaluation_type, output_path):

    if evaluation_type == "teacher":
        all_keywords = ["director", "film", "movie", "cinema", "award", "production", "screenwriter", "cinematographer"]
    elif evaluation_type == "student":
        all_keywords = ["project", "team", "leadership", "communication", "presentation", "collaboration", "achievement", "learning"]

   
    labels = list(missing_scores.keys())
    values = list(missing_scores.values())
    for keyword in all_keywords:
        if keyword not in labels:
            labels.append(keyword)
            values.append(0)

   
    if "project" in labels:
        idx = labels.index("project")
        values[idx] = 15

    # Radar chart setup
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.plot(angles, [15] * len(angles), color='red', linestyle='--', linewidth=1, label='15-point threshold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['pdf_file']
    evaluation_type = request.form.get('evaluation_type', 'student')

    if evaluation_type not in ['teacher', 'student']:
        return jsonify({'error': 'Invalid evaluation type.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    pdf_content, error = extract_text_from_pdf(file_path)
    if error:
        return jsonify({'error': f'Failed to extract text: {error}'}), 500

    pdf_content = pdf_content or extract_text_with_ocr(file_path)

    keyword_weights = {
        "teacher": {
            "director": 9, "film": 7, "movie": 7, "cinema": 5,
            "award": 10, "production": 15, "screenwriter": 23, "cinematographer": 8
        },
        "student": {
            "project": 15, "team": 10, "leadership": 20, "communication": 10,
            "presentation": 8, "collaboration": 12, "achievement": 25, "learning": 10
        },
    }[evaluation_type]

    pdf_score, missing_scores, keyword_frequency = score_content(pdf_content, keyword_weights)
    context_roles = analyze_context(pdf_content)

    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'evaluation_graph.jpg')
    visualize_evaluation(missing_scores, evaluation_type, graph_path)

    if evaluation_type == "teacher":
        result_message = "Pass" if pdf_score >= 50 else "No Pass"
    elif evaluation_type == "student":
        result_message = "Pass" if pdf_score >= 50 else "Fail"

   
    if "project" not in missing_scores:
        missing_scores["project"] = 15

    return jsonify({
        'score': pdf_score,
        'missing_scores': missing_scores,
        'context_roles': context_roles,
        'graph_path': f'/uploads/evaluation_graph.jpg',
        'result_message': result_message
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
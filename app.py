import os
import pandas as pd
import numpy as np
import time
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for, flash
import docx
import fitz  # PyMuPDF

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define functions to generate embeddings
def generate_bert_embeddings(texts, model_name='bert-base-uncased', max_length=128):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()  # Set the model to evaluation mode
    embeddings = []

    for i in range(0, len(texts), 8):  # Process in batches
        batch_texts = texts[i:i + 8]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)

def compute_vector_quality(embeddings):
    if len(embeddings) < 2:
        return 0
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    mean_similarity = similarity_matrix[upper_triangle_indices].mean()
    return mean_similarity

def compute_retrieval_accuracy(embeddings):
    if len(embeddings) < 2:
        return 0
    similarity_matrix = cosine_similarity(embeddings)
    num_queries = len(embeddings)
    num_relevant = 0
    num_retrieved = 0

    for i in range(num_queries):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]  # Sorted by similarity, highest first
        relevant_indices = sorted_indices[sorted_indices != i]  # Exclude self
        if relevant_indices.size > 0:
            num_relevant += 1
            num_retrieved += np.sum(similarity_matrix[i][relevant_indices] > 0.5)  # Threshold for relevance

    return num_retrieved / num_relevant if num_relevant > 0 else 0

class EmbeddingModel:
    def __init__(self, name, embed_func):
        self.name = name
        self.embed_func = embed_func

    def generate_embeddings(self, texts):
        start_time = time.time()
        embeddings = self.embed_func(texts)
        end_time = time.time()
        time_taken = end_time - start_time
        vector_quality = compute_vector_quality(embeddings)
        retrieval_accuracy = compute_retrieval_accuracy(embeddings)
        return embeddings, time_taken, vector_quality, retrieval_accuracy

# Available BERT models
available_models = {
    "BERT Base Uncased": "bert-base-uncased",
    "BERT Large Uncased": "bert-large-uncased",
    "BERT Multilingual": "bert-base-multilingual-uncased"
}

# Function to process text data from files
def process_text_from_csv(file_content):
    df = pd.read_csv(file_content)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        df['features'] = df[categorical_columns].astype(str).values.tolist()
        texts = df['features'].apply(lambda x: ' '.join(x)).tolist()
        return texts
    return []

def process_text_from_docx(file_content):
    doc = docx.Document(file_content)
    text = ' '.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
    return [text]

def process_text_from_pdf(file_content):
    pdf_document = fitz.open(stream=file_content.read(), filetype="pdf")
    text = ''
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return [text]

# Detect file type and process accordingly
def process_file(file_content, filename):
    if filename.lower().endswith('.csv'):
        return process_text_from_csv(file_content)
    elif filename.lower().endswith('.docx'):
        return process_text_from_docx(file_content)
    elif filename.lower().endswith('.pdf'):
        return process_text_from_pdf(file_content)
    else:
        raise ValueError("Unsupported file type")

# Routes
@app.route('/')
def index():
    return render_template('index.html', models=available_models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    model_choice = request.form.get("model_choice")
    selected_model_name = model_choice
    selected_model_identifier = available_models.get(model_choice)

    if model_choice and not selected_model_identifier:
        flash('Invalid model choice')
        return redirect(request.url)

    try:
        texts = process_file(file, file.filename)

        if not texts:
            flash("No text data found in the file.")
            return redirect(request.url)

        models = {
            "BERT Base Uncased": "bert-base-uncased",
            "BERT Large Uncased": "bert-large-uncased",
            "BERT Multilingual": "bert-base-multilingual-uncased"
        }

        results = []
        for model_name, model_identifier in models.items():
            embedding_model = EmbeddingModel("BERT", lambda texts, mi=model_identifier: generate_bert_embeddings(texts, model_name=mi))
            embeddings, time_taken, vector_quality, retrieval_accuracy = embedding_model.generate_embeddings(texts)
            results.append({
                'model_name': model_name,
                'time': time_taken,
                'vector_quality': vector_quality,
                'retrieval_accuracy': retrieval_accuracy
            })
        
        # Determine the recommended model based on the minimum time taken
        recommended_model = min(results, key=lambda x: x['time']) if results else None

        return render_template('results.html', results=results, selected_model_name=selected_model_name, recommended_model=recommended_model)

    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)

import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

nltk.download('punkt')

def get_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return nltk.sent_tokenize(text)

def compute_similarity(sentences):
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix

def generate_color_map(similarity_matrix):
    max_sim = np.max(similarity_matrix)
    min_sim = np.min(similarity_matrix)
    normalized_sim = (similarity_matrix - min_sim) / (max_sim - min_sim)
    return normalized_sim

def map_similarity_to_color(similarity_score):
    cmap = plt.get_cmap("coolwarm")
    color = cmap(similarity_score)
    return rgb2hex(color[:3])

def highlight_sentences(sentences, similarity_matrix):
    highlighted_sentences = []
    for i, sentence in enumerate(sentences):
        similarity_score = np.mean(similarity_matrix[i])
        color = map_similarity_to_color(similarity_score)
        highlighted_sentences.append(f'<span style="background-color:{color}">{sentence}</span>')
    return highlighted_sentences

def save_output(sentences, output_path):
    with open(output_path, 'w') as file:
        file.write("<html><body>\n")
        for sentence in sentences:
            file.write(sentence + " ")
        file.write("\n</body></html>")

def main(input_folder, output_file):
    all_sentences = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            sentences = get_sentences_from_file(file_path)
            all_sentences.extend(sentences)
    
    similarity_matrix = compute_similarity(all_sentences)
    highlighted_sentences = highlight_sentences(all_sentences, similarity_matrix)
    save_output(highlighted_sentences, output_file)

if __name__ == "__main__":
    input_folder = "input_texts"
    output_file = "output/highlighted_output.html"
    main(input_folder, output_file)
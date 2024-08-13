# Sentence Similarity Highlighter

## Overview
This tool takes a set of text files as input, calculates the similarity between sentences, and highlights them in an HTML file. The color similarity between sentences corresponds to their textual similarity.

## How It Works
1. The tool reads sentences from text files.
2. It calculates sentence similarity using TF-IDF vectorization and cosine similarity.
3. Sentences are highlighted based on similarity, with similar sentences having similar background colors.

## Requirements
- Python 3.x
- nltk
- scikit-learn
- matplotlib

## Usage
1. Clone this repository.
2. Place your text files in the `input_texts/` directory.
3. Run the script:
   ```bash
   python similarity_highlighter.py

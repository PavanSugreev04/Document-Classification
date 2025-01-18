<h1 align="center" id="title">Document Classifier</h1>

## Overview
Document Classification with BERTopic and Streamlit is a project designed to automatically label documents based on the textual content present near key areas of interest. This repository demonstrates the development of a document classification system leveraging self-supervised learning with BERTopic and an interactive UI built using Streamlit.

![Project Overview Image](https://github.com/PavanSugreev04/Document-Classification/blob/main/UI.png)

## Project Highlights
- **Self-Supervised Learning with BERTopic**: Utilize BERTopic, a topic modeling algorithm, to group and classify documents based on semantic similarity.
- **Streamlit for Interactive UI**: Provide a user-friendly interface to upload documents, view classifications, and interact with the model in real-time.
- **Flexibility and Adaptability**: Easily extend the system to handle new, unlabeled datasets without requiring extensive retraining.
- **Efficient Document Labeling**: Automatically classify documents into relevant categories based on context, reducing manual effort.

## How it Works
### Text Extraction
- Text content is extracted from documents, with a focus on regions near key information areas.
- Preprocessed text serves as input for further analysis.

### Topic Modeling with BERTopic
- BERTopic is trained on document text to generate embeddings and cluster documents based on semantic similarity.
- Fine-tuning ensures that the model captures domain-specific nuances for classification.

### Classification and Labeling
- Trained BERTopic model assigns topic labels to documents.
- Labels can be mapped to predefined categories for better interpretability.

### Interactive UI
- Streamlit application allows users to:
  - Upload documents in PDF format.
  - View extracted text and predicted labels.
  - Explore clustering insights through interactive visualizations.

## Prerequisites
- **Python**: 3.x (Tested with Python 3.10.12 on Ubuntu 22.04)
- **Libraries**: `bertopic`, `streamlit`, `pandas`, `scikit-learn`, `nltk`, `PyPDF2`
- **Tools**: 
  - Streamlit for the UI
  - Dependencies listed in `requirements.txt`

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PavanSugreev04/Document-Classification.git
   cd document-classification-bertopic
   ```
2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Prepare Data**:
   - Place your training documents in the `data/` directory.
   - Preprocess text using the scripts provided.

4. **Train BERTopic Model**:
   - Use `train_model.py` to train the BERTopic model on your dataset.

5. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```

6. **Interact with the App**:
   - Upload PDF documents.
   - View extracted text, clusters, and topic labels.

## Challenges & Remedies
### Computational Constraints
- **Challenge**: Training BERTopic on large datasets requires significant resources.
- **Remedy**: Utilize cloud-based GPUs or pre-train on smaller datasets and fine-tune as needed.

### Preprocessing Overhead
- **Challenge**: Preprocessing large document sets can be time-intensive.
- **Remedy**: Batch process documents and save intermediate results for reuse.

### Data Quality
- **Challenge**: Inconsistent or sparse text data affects model performance.
- **Remedy**: Apply data augmentation and preprocessing to enhance data quality.

### Hyperparameter Tuning
- **Challenge**: Optimizing BERTopic parameters can be complex.
- **Remedy**: Use grid search or automated tools to explore parameter configurations.

## Data Source
Kaggal News DataSet

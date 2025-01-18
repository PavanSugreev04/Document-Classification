import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import pandas as pd
import io
import tempfile

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.getvalue().decode("utf-8")

def classify_document(text, model, sentence_model):
    """Classify document using BERTopic"""
    # Embed the document
    embeddings = sentence_model.encode([text])
    
    # Get the topic
    topic, proba = model.transform([text])
    
    return topic[0], model.get_topic(topic[0])

def main():
    st.title("Document Topic Classification")
    st.write("Upload your document to classify its topic")

    # Initialize models
    @st.cache_resource
    def load_models():
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        topic_model = BERTopic(embedding_model=sentence_model)
        # You would typically load a pre-trained model here
        # For demonstration, we'll train it on some example data
        return sentence_model, topic_model

    sentence_model, topic_model = load_models()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['txt', 'pdf', 'docx'],
        help="Upload a document to classify"
    )

    if uploaded_file is not None:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract text based on file type
            status_text.text("Extracting text from document...")
            progress_bar.progress(25)
            
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                text = extract_text_from_docx(uploaded_file)
            else:  # txt
                text = extract_text_from_txt(uploaded_file)

            progress_bar.progress(50)
            status_text.text("Classifying document...")

            # Classify document
            topic_id, topic_info = classify_document(text, topic_model, sentence_model)
            
            progress_bar.progress(75)
            status_text.text("Generating results...")

            # Display results
            st.subheader("Classification Results")
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Document Information**")
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Topic ID: {topic_id}")
            
            with col2:
                st.markdown("**Topic Details**")
                if topic_info:
                    words, weights = zip(*topic_info)
                    topic_df = pd.DataFrame({
                        'Word': words,
                        'Weight': weights
                    })
                    st.dataframe(topic_df)
                else:
                    st.write("No specific topic detected")

            # Show document preview
            with st.expander("Document Preview"):
                st.text(text[:500] + "..." if len(text) > 500 else text)

            progress_bar.progress(100)
            status_text.text("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()

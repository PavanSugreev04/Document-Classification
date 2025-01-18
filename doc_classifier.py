import streamlit as st
import pandas as pd
import os
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from bertopic import BERTopic

classifier_model = BERTopic.load("my_11topicmodel")

TOPIC_NAMES = ["New Topic | Outlier", "Technology", "Medical", "Sports",
               "Politics", "Graphics", "Space", "Entertainment",
               "Historical/War", "Food", "History/Egypt"]

def get_file_size(size_bytes):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024

def get_text_from_pdf(file):
    try:
        reader = PdfReader(BytesIO(file.read()))
        all_pages_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            all_pages_text += page.extract_text()
        return all_pages_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise ValueError(e)
    
def get_text_from_txt(file):
    try:
        text = file.read().decode("utf-8")
        return text
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        raise ValueError(e)
    
def get_text_from_docx(file):
    try:
        doc = Document(BytesIO(file.read()))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        raise ValueError(e)

def get_topic_prediction(input_text, model):
   
   topic_id, probability = model.transform(input_text)

   topic_name = TOPIC_NAMES[topic_id[0]+1]
   confidence = probability[0] * 100

   result = {
      "category": topic_name,
      "confidence": confidence
   }

   return result

def get_prediction(document_text):
   
   classification_result = get_topic_prediction(document_text, classifier_model)
   print(classification_result)
   return classification_result

def get_topic(file):
    """Get topic prediction"""
    text = ""

    # Determine file type and extract text
    if file.type == 'application/pdf':
        print("Extracting pdf text...")
        text = get_text_from_pdf(file)
        print(f"Text: {text}")
    elif file.type in ['text/plain', '']:
        print("Extracting txt text...")
        text = get_text_from_txt(file)
        print(f"Text: {text}")
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        print("Extracting docx text...")
        text = get_text_from_docx(file)
        print(f"Text: {text}")
    else:
        return "Unsupported File Type"
    
    prediction = get_prediction(text)
    return prediction

st.set_page_config(layout="wide")
st.title("Document Topic Classifier")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    # Create a list to store file information
    file_data = []
    
    for file in uploaded_files:
        # Get file information
        classification = get_topic(file)
        file_size = get_file_size(file.size)
        print(file_size)
        file_info = {
            'Name': file.name,
            'Size': file_size,
            'Type': file.type or f'.{file.name.split(".")[-1]}',
            'Topic': classification.get("category") if not file_size.startswith("0.0") else "Empty File",
            "Confidence": str(round(classification.get("confidence"),2)) if not file_size.startswith("0.0") else "N/A"
        }
        file_data.append(file_info)
    
    # Create and display dataframe
    df = pd.DataFrame(file_data)
    st.dataframe(
        df,
        column_config={
            "Name": st.column_config.TextColumn("File Name", width="medium"),
            "Size": st.column_config.TextColumn("Size", width="small"),
            "Type": st.column_config.TextColumn("File Type", width="medium"),
            "Topic": st.column_config.TextColumn("Topic", width="medium"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small")
        },
        hide_index=True
    )

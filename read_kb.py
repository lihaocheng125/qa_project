import os
from docx import Document

def read_knowledge_base(folder_path):
    knowledge_base = ""
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.docx'):
                    doc = Document(file_path)
                    for para in doc.paragraphs:
                        knowledge_base += para.text + "\n"
    except Exception as e:
        print(f"Error reading knowledge base: {e}")
    return knowledge_base
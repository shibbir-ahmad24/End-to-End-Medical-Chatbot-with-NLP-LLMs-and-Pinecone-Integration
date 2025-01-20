# End-to-End Medical Chatbot with NLP, LLMs, and Pinecone Integration

## **Problem**
Access to timely and accurate medical information can be challenging for many people. Traditional methods of addressing medical inquiries are often time-consuming and may lack personalization or efficiency.

## **Solution**
An end-to-end **Medical Chatbot** that leverages cutting-edge technologies to provide accurate, real-time responses to user queries about symptoms, medications, and general health topics.

Key components include:
- **Large Language Models (LLMs)** for intelligent and natural conversations.
- **Hugging Face** and **Langchain** for seamless NLP integration.
- **Pinecone Vector Database** for efficient storage and retrieval of embeddings.
- A user-friendly web application built with Flask, HTML, and CSS for accessibility.

## **Impact**
- **Improved Healthcare Accessibility**: Provides instant, accurate medical information to users.
- **Enhanced User Experience**: Delivers personalized and context-aware responses through advanced NLP techniques.
- **Educational Growth**: Encourages understanding of modern AI techniques, vector databases, and web app development.

## **Tech Stack**
- **Languages**: Python, HTML, CSS
- **Frameworks**: Flask, Langchain
- **Models**: Llama-2-7b-chat, Hugging Face sentence-transformers
- **Database**: Pinecone Vector Database
- **Tools**: Hugging Face, Langchain, Embedding Models

## **Summary**
- Developed an **end-to-end Medical Chatbot** using Python and state-of-the-art tools.
- Integrated **LLMs** for intelligent and natural language processing.
- Utilized **Hugging Face** and **Langchain** for advanced NLP capabilities.
- Implemented **Pinecone Vector Database** for efficient embedding storage and retrieval.
- Designed a responsive web app with **Flask**, **HTML**, and **CSS**.
- Learned and applied concepts like embedding models, vector databases, and chatbot frameworks.

## **Steps to Run the Project**

Follow these steps to set up and run the Medical Chatbot project:

1. **Create a Conda Environment:**
   ```
   conda create -n mchatbot python=3.11 -y
   ```
2. **Activate the Environment:**
   ```
    activate mchatbot
   ```
3. Install Required Dependencies:
   ```
   pip install -r requirements.txt
   ```
   
## **Code Snippet**

### **Text Chunking Function**
This function splits the extracted text data into manageable chunks for better processing and analysis.

```python
### **Create text chunks**
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
```

### **Download Embedding Model**
```python
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

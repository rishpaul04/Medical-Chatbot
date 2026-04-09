from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)
load_dotenv()

# Load API Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Ensure keys are set in environment if needed by libraries implicitly
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



# Initialize Embeddings
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embedding = download_embeddings()

# Initialize Pinecone and Retriever
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,  
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})    

# Initialize Chat Model
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
)

# Initialize Chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input_text = msg
    print(input_text)
    
    response = rag_chain.invoke({"input": input_text})
    print("Response:", response["answer"])
    
    return str(response["answer"])   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
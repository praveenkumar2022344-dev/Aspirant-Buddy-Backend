import os
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import zipfile
import chromadb
import google.generativeai as genai

# Unzip database if it exists as a zip but not as a folder
if not os.path.exists("chroma_db") and os.path.exists("chroma_db.zip"):
    print("Unzipping chroma_db...")
    with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("❌ Error: Please set your GEMINI_API_KEY in the .env file!")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="aspirant_knowledge")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_buddy(request: QuestionRequest):
    question = request.question
    
    # Search database for relevant context
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    context = ""
    if results['documents'] and results['documents'][0]:
        context = "\n".join(results['documents'][0])
        
    prompt = f"""
    You are 'Aspirant Buddy', an expert mentor for students (like IIT JEE/NEET aspirants).
    You speak casually in Hinglish (Hindi + English). 
    Use the following 'Study Material & Interviews Context' to answer the user's question. 
    If the context doesn't contain the answer, give a helpful generic answer but mention that it's your own advice.
    Keep your answer conversational, motivating, and strictly under 3 sentences so it's easy to hear.
    Do not use complex formatting or bullet points. Just plain text.
    
    Context from YouTube Interviews: {context}
    
    Student Question: {question}
    
    Aspirant Buddy (Hinglish response):
    """
    
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('*', '').replace('#', '')
        return {"answer": clean_text}
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return {"answer": "Bhaiya thoda technical error aa raha hai, please baad mein try karna."}

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so the phone can access it over WiFi
    print("🚀 Starting Aspirant Buddy Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

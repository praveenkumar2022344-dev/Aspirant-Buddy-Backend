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
from fastapi.responses import FileResponse
import subprocess

# Unzip database if it exists as a zip but not as a folder
if not os.path.exists("chroma_db") and os.path.exists("chroma_db.zip"):
    print("Unzipping chroma_db...")
    with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="aspirant_knowledge")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_buddy(request: QuestionRequest):
    question = request.question
    results = collection.query(query_texts=[question], n_results=3)
    context = ""
    if results['documents'] and results['documents'][0]:
        context = "\n".join(results['documents'][0])
        
    prompt = f'''
    You are 'Aspirant Buddy', an expert mentor for students (like IIT JEE/NEET aspirants).
    You speak casually in Hinglish (Hindi + English). 
    Use the following 'Study Material & Interviews Context' to answer the user's question. 
    If the context doesn't contain the answer, give a helpful generic answer but mention that it's your own advice.
    Keep your answer conversational, motivating, strictly under 3 sentences. No formatting.
    
    Context from YouTube Interviews: {context}
    Student Question: {question}
    Aspirant Buddy (Hinglish response):
    '''
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('*', '').replace('#', '')
        return {"answer": clean_text}
    except Exception as e:
        return {"answer": "Bhaiya thoda technical error aa raha hai, please baad mein try karna."}

class SpeakRequest(BaseModel):
    text: str

@app.post("/speak")
def speak(request: SpeakRequest):
    text = request.text
    output_file = "response.mp3"
    try:
        # Premium Indian Voice (Madhur)
        subprocess.run(["edge-tts", "--voice", "hi-IN-MadhurNeural", "--text", text, "--write-media", output_file], check=True)
        return FileResponse(output_file, media_type="audio/mpeg")
    except Exception as e:
        return {"error": str(e)}

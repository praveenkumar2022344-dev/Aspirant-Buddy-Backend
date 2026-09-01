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
from google import genai
from google.genai import types
from fastapi.responses import FileResponse
import wave

if not os.path.exists("chroma_db") and os.path.exists("chroma_db.zip"):
    print("Unzipping chroma_db...")
    with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt
        )
        clean_text = response.text.replace('*', '').replace('#', '')
        return {"answer": clean_text}
    except Exception as e:
        return {"answer": "Bhaiya thoda technical error aa raha hai, please baad mein try karna."}

class SpeakRequest(BaseModel):
    text: str

@app.post("/speak")
def speak(request: SpeakRequest):
    text = request.text
    output_file = "response.wav"
    try:
        # Directly hitting Gemini Live TTS Engine
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',
            contents=f'Only generate audio from this exact transcript: {text}',
            config=types.GenerateContentConfig(response_modalities=["AUDIO"])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Converting Raw Audio to Phone playable Format
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(part.inline_data.data)
                return FileResponse(output_file, media_type="audio/wav")
        return {"error": "No audio generated"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

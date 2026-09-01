import os
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
import zipfile
import chromadb
from google import genai
from google.genai import types
import re

if not os.path.exists("chroma_db") and os.path.exists("chroma_db.zip"):
    print("Unzipping chroma_db...")
    with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="aspirant_knowledge")

app = FastAPI()

async def generate_sentence_audio(text, websocket):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',
            contents=f'Only generate audio from this exact transcript: {text}',
            config=types.GenerateContentConfig(response_modalities=["AUDIO"])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                await websocket.send_bytes(part.inline_data.data)
    except Exception as e:
        print("Audio Error:", e)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        question = await websocket.receive_text()
        
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
        
        response = client.models.generate_content_stream(
            model='gemini-3.5-flash',
            contents=prompt
        )
        
        sentence_buffer = ""
        for chunk in response:
            if chunk.text:
                await websocket.send_text(chunk.text)
                sentence_buffer += chunk.text
                
                parts = re.split(r'([.!?\n]+)', sentence_buffer)
                if len(parts) > 1:
                    for i in range(0, len(parts)-1, 2):
                        sentence = parts[i] + parts[i+1]
                        if sentence.strip():
                            await generate_sentence_audio(sentence.strip(), websocket)
                    sentence_buffer = parts[-1]
        
        if sentence_buffer.strip():
             await generate_sentence_audio(sentence_buffer.strip(), websocket)
             
        await websocket.send_text("[DONE]")
    except Exception as e:
        print("WS Error:", e)
    finally:
        await websocket.close()

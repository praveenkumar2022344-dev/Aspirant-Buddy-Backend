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

if not os.path.exists('chroma_db') and os.path.exists('chroma_db.zip'):
    print('Unzipping chroma_db...')
    with zipfile.ZipFile('chroma_db.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='aspirant_knowledge')

app = FastAPI()

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        question = await websocket.receive_text()
        
        results = collection.query(query_texts=[question], n_results=3)
        context = ''
        if results['documents'] and results['documents'][0]:
            context = chr(10).join(results['documents'][0])
            
        prompt = 'You are Aspirant Buddy, an expert mentor for students (like IIT JEE/NEET aspirants).\nYou speak casually in Hinglish (Hindi + English).\nUse the following Study Material & Interviews Context to answer the user question.\nIf the context does not contain the answer, give a helpful generic answer but mention that it is your own advice.\nKeep your answer conversational, motivating, strictly under 3 sentences. No formatting.\n\nContext from YouTube Interviews: ' + context + '\nStudent Question: ' + question + '\nAspirant Buddy (Hinglish response):'
        
        # Step 1: Text ko live stream karo (Speed ke liye)
        response = client.models.generate_content_stream(
            model='gemini-3.5-flash',
            contents=prompt
        )
        
        full_text = ''
        for chunk in response:
            if chunk.text:
                await websocket.send_text(chunk.text)
                full_text += chunk.text
                
        # Step 2: Ek hi baari me audio download karo (Limit se bachne ke liye)
        if full_text.strip():
            try:
                audio_response = client.models.generate_content(
                    model='gemini-2.5-flash-preview-tts',
                    contents=f'Only generate audio from this exact transcript: {full_text.strip()}',
                    config=types.GenerateContentConfig(response_modalities=['AUDIO'])
                )
                for part in audio_response.candidates[0].content.parts:
                    if part.inline_data:
                        await websocket.send_bytes(part.inline_data.data)
            except Exception as e:
                print('Audio Error:', e)
             
        await websocket.send_text('[DONE]')
    except Exception as e:
        print('WS Error:', e)
    finally:
        await websocket.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

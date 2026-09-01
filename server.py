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
import requests
import base64

if not os.path.exists('chroma_db') and os.path.exists('chroma_db.zip'):
    print('Unzipping chroma_db...')
    with zipfile.ZipFile('chroma_db.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='aspirant_knowledge')

app = FastAPI()

# --- KEY ROTATOR LOGIC ---
# Aap future me yahan neeche aur keys add kar sakte hain
SARVAM_KEYS = [
    "sk_v2ex5d01_JVtxZhhoxSxyf0DBtyLnoaXK",
    "sk_aack2civ_4uPEkttLv8nA8Ka7IcYQCLAC",
    "sk_ykpy29fl_kvBzBOoyZdzhIu1RDaleXUTz",
    "sk_1apyhpnb_K4QrxO6GmJksKI8YmGvTppQg",
    "sk_g5ey27q5_A1zPvDLgfYwp9CldWAzCBSrP",
    "sk_maa0tann_shtZtWIBVT9egvYKMZ6dYnFQ",
    "sk_xo7163vs_RLKovrCIeqyj01ZXGIH2xkT6"
]
current_key_index = 0

def get_sarvam_audio(text: str):
    global current_key_index
    url = 'https://api.sarvam.ai/text-to-speech'
    
    # Ye loop tab tak chalega jab tak koi na koi key kaam kar jaye
    for _ in range(len(SARVAM_KEYS)):
        current_key = SARVAM_KEYS[current_key_index]
        payload = {
            'inputs': [text],
            'target_language_code': 'hi-IN',
            'speaker': 'shubh', # Male voice
            'pace': 1.0,
            'speech_sample_rate': 24000,
            'enable_preprocessing': True,
            'model': 'bulbul:v3'
        }
        headers = {
            'api-subscription-key': current_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return base64.b64decode(data['audios'][0])
            else:
                print(f"Key {current_key_index} limit over/failed. Error: {response.text}")
                # Switch to next key magically
                current_key_index = (current_key_index + 1) % len(SARVAM_KEYS)
                print(f"Switching to next key {current_key_index}")
        except Exception as e:
            print(f"Request failed: {e}")
            current_key_index = (current_key_index + 1) % len(SARVAM_KEYS)
            
    return None

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
        
        response = client.models.generate_content_stream(
            model='gemini-3.5-flash',
            contents=prompt
        )
        
        full_text = ''
        for chunk in response:
            if chunk.text:
                await websocket.send_text(chunk.text)
                full_text += chunk.text
                
        if full_text.strip():
            # Yahan text pura hone ke baad audio aayega Sarvam API se
            audio_bytes = get_sarvam_audio(full_text.strip())
            if audio_bytes:
                # 44 bytes ka WAV header skip karte hain taaki click/noise na aaye
                pcm_bytes = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                await websocket.send_bytes(pcm_bytes)
             
        await websocket.send_text('[DONE]')
    except Exception as e:
        print('WS Error:', e)
    finally:
        await websocket.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

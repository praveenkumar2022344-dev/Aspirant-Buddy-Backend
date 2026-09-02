import os
import sys
import asyncio
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

# Note: ChromaDB completely disabled to prevent Render 512MB RAM Crash (OOMKill)
# chroma_client = chromadb.PersistentClient(path='./chroma_db')
# collection = chroma_client.get_or_create_collection(name='aspirant_knowledge')

app = FastAPI()

# --- KEY ROTATOR LOGIC ---
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

def get_sarvam_audio_sync(text: str):
    global current_key_index
    url = 'https://api.sarvam.ai/text-to-speech'
    
    for _ in range(len(SARVAM_KEYS)):
        current_key = SARVAM_KEYS[current_key_index]
        payload = {
            'inputs': [text],
            'target_language_code': 'hi-IN',
            'speaker': 'shubh', 
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
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return base64.b64decode(data['audios'][0])
            elif response.status_code == 400:
                # Text length error (e.g. >500 char), ignore without burning key
                print(f"Text length error or bad request: {response.text}")
                return None
            else:
                print(f"Key {current_key_index} limit over/failed. Error: {response.text}")
                current_key_index = (current_key_index + 1) % len(SARVAM_KEYS)
                print(f"Switching to next key {current_key_index}")
        except Exception as e:
            print(f"Request failed: {e}")
            current_key_index = (current_key_index + 1) % len(SARVAM_KEYS)
            
    return None

from starlette.concurrency import run_in_threadpool

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        question = await websocket.receive_text()
        
        # 1. Database Query (Disabled to prevent Render 512MB RAM Crash)
        context = ''
        
        # NAYA PROMPT JO INTENT SAMAJHEGA
        prompt = 'You are Aspirant Buddy, an expert mentor for students (like IIT JEE/NEET aspirants).\nYou speak casually in Hinglish (Hindi + English).\nUse the following Study Material & Interviews Context to answer the user question.\nIf the context does not contain the answer, give a helpful generic answer but mention that it is your own advice.\nAdjust the length of your answer based on the intent and complexity of the question. For casual talk or simple questions, keep it very short (1-3 sentences). For complex topics, deep questions, or when explanation is naturally needed, give a detailed and longer response automatically. No markdown formatting.\n\nContext from YouTube Interviews: ' + context + '\nStudent Question: ' + question + '\nAspirant Buddy (Hinglish response):'
        
        # 2. Start Gemini AI stream
        try:
            response = await client.aio.models.generate_content_stream(
                model='gemini-flash-latest',
                contents=prompt
            )
        except Exception as api_err:
            error_msg = str(api_err).lower()
            if '429' in error_msg or 'quota' in error_msg:
                await websocket.send_text("Bhaiya thoda dheere! Google ne 1 minute me jyada sawal puchhne par limit laga di hai. 1 minute wait karke puchhiye.")
            else:
                await websocket.send_text("Google API me kuch dikkat aayi hai. Wapas try karo.")
            await websocket.send_text("[DONE]")
            return
            
        # 3. Audio Worker Queue
        sentence_queue = asyncio.Queue()
        
        async def audio_worker():
            while True:
                text_chunk = await sentence_queue.get()
                if text_chunk is None:
                    break
                # Force split if > 450 to prevent Sarvam 500 char limit error
                chunks_to_process = []
                while len(text_chunk) > 450:
                    chunks_to_process.append(text_chunk[:450])
                    text_chunk = text_chunk[450:]
                if text_chunk.strip():
                    chunks_to_process.append(text_chunk.strip())
                    
                for c in chunks_to_process:
                    audio = await run_in_threadpool(get_sarvam_audio_sync, c)
                    if audio:
                        pcm = audio[44:] if len(audio)>44 else audio
                        await websocket.send_bytes(pcm)
                        
                sentence_queue.task_done()

        worker_task = asyncio.create_task(audio_worker())
        
        import re
        current_sentence = ''
        
        try:
            async for chunk in response:
                if chunk.text:
                    await websocket.send_text(chunk.text)
                    current_sentence += chunk.text
                    
                    # Check for punctuation, including commas to break up long sentences
                    match = re.search(r'[,.?!।]\s|\n', current_sentence)
                    if match:
                        split_idx = match.end()
                        sentence_to_speak = current_sentence[:split_idx].strip()
                        current_sentence = current_sentence[split_idx:]
                        
                        if len(sentence_to_speak) > 2:
                            await sentence_queue.put(sentence_to_speak)
        except Exception as api_err:
            print("Gemini stream error:", api_err)
            await websocket.send_text(" Google API thodi slow chal rahi hai, wapas try karo.")
        
        if current_sentence.strip():
            await sentence_queue.put(current_sentence.strip())
            
        await sentence_queue.put(None)
        await worker_task
             
        await websocket.send_text('[DONE]')
    except Exception as e:
        print('WS Error:', e)
    finally:
        await websocket.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

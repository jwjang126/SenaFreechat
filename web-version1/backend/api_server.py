import sys
import os
import json
import subprocess
import whisper
from TTS.api import TTS
import logging
import torch
from transformers import pipeline
import uvicorn
import base64
import uuid # 임시 파일 이름 생성을 위해 추가

# FastAPI 관련 라이브러리 import
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

# --- 이 부분은 process_audio.py와 동일합니다 ---
# --- 서버 시작 시 단 한번만 실행됩니다! ---
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s:%(name)s:%(message)s')

pipe = None
whisper_model = None
tts_model = None
try:
    print("Loading AI models... This might take a moment.", file=sys.stderr)
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map="auto")
    pipe.tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
        "{% endfor %}"
        "{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
    )
    whisper_model = whisper.load_model("base")
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    print("All models loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"Failed to load models: {e}", file=sys.stderr)
# -----------------------------------------------------------

# FastAPI 앱 생성
app = FastAPI()

# '/process-audio' 경로로 POST 요청을 처리할 API 엔드포인트 생성
@app.post("/process-audio/")
async def process_audio(
    audio: UploadFile = File(...),
    conversationHistory: str = Form(...),
    topic: str = Form(...),
    level: str = Form(...)
):
    # 임시 파일 경로들 생성
    temp_dir = "uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    unique_filename = str(uuid.uuid4())
    input_webm_path = os.path.join(temp_dir, f"{unique_filename}.webm")
    input_wav_path = os.path.join(temp_dir, f"{unique_filename}_16k.wav")
    output_mp3_path = os.path.join(temp_dir, f"{unique_filename}.mp3")

    try:
        # 1. 업로드된 webm 파일을 디스크에 저장
        with open(input_webm_path, "wb") as buffer:
            buffer.write(await audio.read())

        # 2. webm -> wav 변환 (ffmpeg 사용)
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_webm_path, "-ac", "1", "-ar", "16000", "-f", "wav", input_wav_path],
            check=True, capture_output=True, text=True
        )

        # ------------------------------------------------
        # 아래는 기존 process_audio.py의 main 함수 로직과 거의 동일
        conversation_history = json.loads(conversationHistory) if conversationHistory else []

        # STT (Whisper)
        result = whisper_model.transcribe(input_wav_path)
        user_text = result.get("text", "").strip()

        # LLM 호출
        system_prompt_content = f"""You are 'Tutor', a friendly English conversation partner.
Today's conversation topic is: {topic}.
Please adjust your language difficulty to a {level} level for the student.
... (이하 프롬프트 내용은 동일)
"""
        system_prompt = {"role": "system", "content": system_prompt_content}
        messages = [system_prompt] + conversation_history + [{"role": "user", "content": user_text}]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(
            prompt, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95,
            repetition_penalty=1.2  # 👈 이 옵션을 추가하세요! (1.0보다 큰 값)
        )
        ai_reply = outputs[0]["generated_text"][len(prompt):].strip()
        ai_reply = ai_reply.replace("rbrakkendcode", "").strip()

        # TTS 및 MP3 변환
        temp_tts_wav = os.path.join(temp_dir, f"temp_tts_{unique_filename}.wav")
        tts_model.tts_to_file(text=ai_reply, file_path=temp_tts_wav)
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_tts_wav, output_mp3_path],
            check=True, capture_output=True, text=True
        )
        os.remove(temp_tts_wav)
        # ------------------------------------------------

        # 3. 생성된 mp3 파일을 base64로 인코딩하여 JSON으로 반환
        with open(output_mp3_path, "rb") as f:
            mp3_bytes = f.read()
            audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
        
        return JSONResponse(content={
            "user_text": user_text,
            "ai_reply": ai_reply,
            "audio_base64": audio_base64 # mp3 파일을 텍스트로 인코딩하여 전달
        })

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 4. 모든 임시 파일 삭제
        for f_path in [input_webm_path, input_wav_path, output_mp3_path]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e:
                    print(f"Error removing file {f_path}: {e}", file=sys.stderr)

# 이 파일을 직접 실행할 경우 uvicorn 서버를 8000번 포트로 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
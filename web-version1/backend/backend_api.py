# backend_api.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import numpy as np
import soundfile as sf
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 개발용: React 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 모델 & TTS 로드
# -------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC",
          progress_bar=False,
          gpu=torch.cuda.is_available())

# -------------------------------
# 세션 상태 (메모리) - 간단 예시
# -------------------------------
conversation = []
conversation_summary = ""
MAX_CONTEXT = 10

# -------------------------------
# STT 함수
# -------------------------------
def run_stt(audio: np.ndarray, fs: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    import whisper
    whisper_model = whisper.load_model("base", device="cpu")
    result = whisper_model.transcribe(tmp.name, task="transcribe", language="en", temperature=0)
    return result.get("text", "").strip()

# -------------------------------
# LLM 답변 생성
# -------------------------------
def generate_ai_response(user_input: str, level: str = "중급") -> str:
    global conversation, conversation_summary
    recent_convos = conversation[-MAX_CONTEXT:]

    context_text = ""
    if conversation_summary:
        context_text += f"[Summary of previous conversation]\n{conversation_summary}\n\n"

    for turn in recent_convos:
        context_text += f"Student: {turn['user']}\nTutor: {turn.get('ai','')}\n"

    sentence_count = {"초급": 2, "중급": 4, "고급": 6}[level]

    prompt = f"""
You are a friendly English conversation tutor.
Your Persona:
- You are friendly, encouraging, and patient.
- Your goal is to make the student feel comfortable and encourage them to speak more.

Conversation Rules:
1.  **Be expressive:** Your response should be around 3-4 complete sentences.
2.  **Always ask a question:** You MUST end your response with a question to keep the conversation going.
3.  **Paraphrase, Don't Repeat:** You MUST NEVER repeat the student's sentences or key phrases.
4.  **Stay in Your Role:** You are the 'Tutor'. You MUST NOT write the "Student:" part of the conversation.
5.  Respond naturally according to the student's level.
6.  End sentences properly: All sentences must end with proper punctuation (., !, ?).

The student has provided the following topic: "{user_input}".

Level guidance:
- Difficulty: {level}
- Student description:
    - 초급: Think the student is a high school student. Use simple words and short sentences.
    - 중급: Think the student is an adult learner. Use moderate vocabulary and clear explanations.
    - 고급: Think the student wants to speak like a native speaker. Use advanced vocabulary and natural expressions.
- Expected response length: around {sentence_count} sentences.


---
**Examples of what to do and what NOT to do:**

**(O) Good Example - Asks a question:**
Me: "I enjoyed watching a movie yesterday."
You: "That sounds like a fun way to relax! It's great to take a break sometimes. What kind of film was it?"

**(X) Bad Example - Does NOT ask a question:**
Me: "I like listening to music."
You: "That's a great hobby. Music can be very relaxing."
**(This is bad because it doesn't ask a question and ends the conversation.)**

**(O) Good Example - Ends properly and asks a question:**
Me: "I enjoyed watching a movie yesterday."
You: "That sounds like a fun way to relax! It's great to take a break sometimes. What kind of film was it?"

**(X) Bad Example - Does not end properly:**
Me: "I like listening to music."
You: "That's a great hobby. Music can be very relaxing"

Send only the answer of "You" in print.
---

Context so far:
{context_text}

Tutor:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Tutor:" in reply:
        reply = reply.split("Tutor:")[-1].strip()

    conversation.append({"user": user_input, "ai": reply})

    # 요약 갱신
    if len(conversation) % MAX_CONTEXT == 0:
        summary_prompt = f"Summarize the following conversation briefly:\n{context_text}"
        summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to(DEVICE)
        summary_outputs = model.generate(**summary_inputs, max_new_tokens=100)
        conversation_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

    return reply

# -------------------------------
# TTS 변환
# -------------------------------
def generate_tts_audio(text: str) -> np.ndarray:
    wav = tts.tts(text=text)
    return wav  # numpy array

# -------------------------------
# FastAPI 정의
# -------------------------------


class ChatRequest(BaseModel):
    text: str
    level: str = "중급"

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    reply = generate_ai_response(req.text, req.level)
    audio_np = generate_tts_audio(reply)
    return {"reply": reply, "audio": audio_np.tolist()}

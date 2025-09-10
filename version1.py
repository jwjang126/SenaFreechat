# 🔊 SENA - 안정화 버튼 기반 녹음 + 대화 로그
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf

# -------------------------------
# 모델 & TTS 로드
# -------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_resource
def load_tts():
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC",
              progress_bar=False,
              gpu=torch.cuda.is_available())
    return tts

tts = load_tts()

# -------------------------------
# 세션 상태 초기화
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# -------------------------------
# 녹음 함수
# -------------------------------
def record_audio(duration=5, fs=16000):
    """duration 초 동안 녹음하고 반환"""
    st.info(f"🔴 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Finished!")
    return audio.flatten(), fs

# -------------------------------
# STT
# -------------------------------
def run_stt(audio, fs):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    import whisper
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(tmp.name, task="transcribe", language="en", temperature=0)
    return result.get("text", "").strip()

# -------------------------------
# LLM 답변 생성
# -------------------------------
def generate_ai_response(user_input):
    conversation_text = ""
    for turn in st.session_state.conversation[-5:]:
        conversation_text += f"Student: {turn['user']}\nTutor: {turn['ai']}\n"

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

---
**Examples of what to do and what NOT to do:**

**(O) Good Example - Asks a question:**
Student: "I enjoyed watching a movie yesterday."
Tutor: "That sounds like a fun way to relax! It's great to take a break sometimes. What kind of film was it?"

**(X) Bad Example - Does NOT ask a question:**
Student: "I like listening to music."
Tutor: "That's a great hobby. Music can be very relaxing."
**(This is bad because it doesn't ask a question and ends the conversation.)**
---

Conversation so far:
{conversation_text}

Student: "{user_input}"
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

    # Tutor: 뒤에 오는 부분만 추출
    if "Tutor:" in reply:
        reply = reply.split("Tutor:")[-1].strip()

    return reply


# -------------------------------
# TTS 재생
# -------------------------------
def speak(text):
    wav = tts.tts(text=text)
    sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔊 SENA - English Tutor")
st.write("🎤 버튼을 눌러 대화하세요!")

# 녹음 버튼
duration = st.slider("발화 길이 (초)", 5, 15, 5)
if st.button("My turn"):
    audio, fs = record_audio(duration, fs=16000)
    # --- 1. 음성인식 중 스피너 표시 ---
    with st.spinner(""):
        user_text = run_stt(audio, fs)
    
    if user_text:
        st.markdown(f"**You:** {user_text}")

        # --- 2. AI 답변 생성 중 스피너 표시 (요청하신 부분) ---
        with st.spinner(""):
            ai_reply = generate_ai_response(user_text)
            st.session_state.conversation.append({"user": user_text, "ai": ai_reply})
        
        st.markdown(f"**AI Tutor:** {ai_reply}")

        # --- 3. TTS 음성 생성 중 스피너 표시 ---
        with st.spinner(""):
            speak(ai_reply)
    else:
        st.warning("Could not recognize any speech. Please try again.")
    user_text = run_stt(audio, fs)
    st.markdown(f"**You:** {user_text}")

    ai_reply = generate_ai_response(user_text)
    st.session_state.conversation.append({"user": user_text, "ai": ai_reply})
    st.markdown(f"**AI Tutor:** {ai_reply}")
    speak(ai_reply)

# 대화 로그
st.markdown("## 💬 대화 내용")
for turn in st.session_state.conversation:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**AI Tutor:** {turn['ai']}")

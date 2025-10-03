# 🔊 SENA - 안정화 버튼 기반 녹음 + 대화 로그
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import time
from io import StringIO
import datetime

# -------------------------------
# 모델 & TTS 로드
# -------------------------------
MODEL_NAME = "google/gemma-2b-it"
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
def record_audio_fixed(duration=5, fs=16000):
    """duration 초 동안 녹음하고 반환 (고정 길이)"""
    st.info(f"🔴 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Finished!")
    return (audio.flatten().astype(np.float32)/32768.0), fs

def record_audio_vad(silence_sec=2.0, fs=16000, silence_thresh=0.01, max_total_sec=15):
    """무음이 silence_sec 이상 지속되면 자동 종료. 안전상 max_total_sec 넘어가면 강제 종료."""
    st.info(f"🔴 Listening... 무음 {silence_sec}초 지속 시 자동 종료 (최대 {max_total_sec}초)")
    frames = []
    block_dur = 0.05
    block_size = int(fs*block_dur)
    silent_run = 0.0
    start = time.time()

    with sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:
        while True:
            block, _ = stream.read(block_size)
            x = block[:, 0]
            frames.append(x.copy())
            rms = float(np.sqrt(np.mean(x**2) + 1e-12))
            silent_run = (silent_run + block_dur) if rms < silence_thresh else 0.0
            if silent_run >= silence_sec:
                break
            if time.time() - start >= max_total_sec:
                break

    audio = np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, dtype=np.float32)
    st.success("✅ Finished!")
    return audio, fs

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
# 이전 대화 요약
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

MAX_CONTEXT = 10

# -------------------------------
# LLM 답변 생성
# -------------------------------
def generate_ai_response(user_input, topic, level, sentence_count):
    recent_convos = st.session_state.conversation[-MAX_CONTEXT:]
    context_text = ""
    if st.session_state.conversation_summary:
        context_text += f"[Summary of previous conversation]\n{st.session_state.conversation_summary}\n\n"

    full_context_list = recent_convos + [{"user": user_input}]
    temp_context_text = ""
    for turn in full_context_list:
        temp_context_text += f"Student: {turn['user']}\n"
        if 'ai' in turn and turn['ai']:
            temp_context_text += f"Tutor: {turn['ai']}\n"
    context_text = temp_context_text

    level_instructions = {
        "초급": "Speak like you are talking to a high school student. Use simple vocabulary and very short sentences (1-2 sentences).",
        "중급": "Use clear, everyday vocabulary for adult learners. Your response should be around 2-3 sentences.",
        "고급": "Use advanced vocabulary and natural, native-like expressions to challenge the student. Your response can be longer, around 3-4 sentences."
    }
    level_instruction = level_instructions[level]

    # --- Final Enhanced Prompt ---
    prompt = f"""
## Persona
You are 'Tutor', an AI English conversation coach. Your personality is warm, endlessly curious, and patient. Your primary goal is to make the student feel comfortable and encourage them to share their own thoughts and stories, much like a friendly chat.

## Core Conversation Rules
1.  **Listen and Engage:** Always start by briefly acknowledging what the student just said.
2.  **Ask Personal, Open-ended Questions:** Your main tool is asking simple, open-ended questions about the student's personal experiences or opinions on the topic: **{topic}**.
3.  **Answering Personal Questions:** If the student asks about your preferences (e.g., "Do you like chicken?"), give a simple, natural answer (e.g., "Oh, I love it!"). Then, immediately ask the student a question to pivot back to them. Never say "I can help you learn".
4.  **Adapt to the Student's Level:** You must follow this specific instruction: **{level_instruction}**

## Unbreakable Rules
1.  **The Question is Key:** Your response **must always** end with a question. There are no exceptions.
2.  **Dialogue Only - No Actions:** This is critical. Your response must **only** contain the words you would speak. Do **not** include any descriptions of actions, emotions, or stage directions, especially anything in asterisks (e.g., *smiles warmly*, *nods*, *focuses on student*).
3.  **Stay in Character:** You are only the 'Tutor'. Never break character to output instructions, code, or anything that isn't part of the conversation.

## Conversation History
{context_text}
Tutor:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.2
    )

    new_tokens = outputs[0, input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if (len(st.session_state.conversation) + 1) % MAX_CONTEXT == 0:
        summary_prompt = f"Summarize the following conversation briefly:\n{context_text}"
        summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to(DEVICE)
        summary_outputs = model.generate(**summary_inputs, max_new_tokens=100)
        st.session_state.conversation_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True).strip()
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
topic = st.text_input("주제를 입력하세요")
level = st.selectbox("수준을 선택하세요", ["초급", "중급", "고급"])

# 수준별 문장 수 설정
sentence_count = {"초급": 2, "중급": 4, "고급": 6}[level]

# -------------------------------
# 녹음 종료 방식 선택
# -------------------------------
st.sidebar.header("🎚️ 녹음 설정")
mode = st.sidebar.radio("발화 종료 방식", ("무음 자동 종료", "고정 길이 제한"), horizontal=True)

if mode == "무음 자동 종료":
    silence_sec = st.sidebar.slider("무음 지속 시간(초)", 1.0, 5.0, 2.0, 0.5, format="%.1f")
    with st.expander("고급 설정 보기"):
        silence_thresh = st.sidebar.slider("무음 임계(RMS x100, 낮을수록 민감)", 0.2, 3.0, 1.0, 0.4, format="%.1f")/100
        safety_cap = st.sidebar.slider("최대 발화 시간(초)", 5, 30, 15)
    st.sidebar.caption(f"💡 연속 무음이 {silence_sec}초 이상이면 자동 종료됩니다. (최대 발화 시간 {safety_cap}초)")
else:
    duration = st.sidebar.slider("발화 길이 제한 (초)", 3, 15, 5)
    st.sidebar.caption(f"💡 총 발화 시간 {duration}초가 지나면 자동으로 종료됩니다.")

st.divider()

# -------------------------------
# 녹음 + STT + LLM + TTS 트리거
# -------------------------------
if st.button("My turn"):
    if mode == "무음 자동 종료":
        audio, fs = record_audio_vad(
            silence_sec=silence_sec,
            fs=16000,
            silence_thresh=silence_thresh,
            max_total_sec=safety_cap
        )
    else:
        audio, fs = record_audio_fixed(duration, fs=16000)

    with st.spinner("📝 인식 중..."):
        user_text = run_stt(audio, fs)
    
    if user_text:
        st.markdown(f"**You:** {user_text}")
        with st.spinner("🤖 답변 생성 중..."):
            ai_reply = generate_ai_response(user_text, topic, level, sentence_count)
            st.session_state.conversation.append({"user": user_text, "ai": ai_reply})
        with st.spinner("🗣️ 발화 재생 중..."):
            speak(ai_reply)
    else:
        st.warning("Could not recognize any speech. Please try again.")


# -------------------------------
# 📄 전체 대화 보기 / 다운로드(.txt)
# -------------------------------
def get_full_transcript_txt():
    """텍스트 형식으로 전체 대화 반환"""
    lines = []
    for turn in st.session_state.conversation:
        if turn.get("user"): lines.append(f"You: {turn['user']}")
        if turn.get("ai"):   lines.append(f"AI Tutor: {turn['ai']}")
        lines.append("-"*40)
    return "\n".join(lines[:-1]) if lines else "(no conversation yet)"

# -------------------------------
# Streamlit UI
# -------------------------------
#st.markdown("##### 📄 전체 대화 보기 / 다운로드")

today_str = datetime.datetime.now().strftime("%y%m%d")
safe_topic = topic.replace(" ", "_") if topic else "NoTopic"
file_name = f"{today_str}_{safe_topic}_{level}.txt"

with st.expander("📄 전체 대화 보기 / 다운로드"):
    st.text_area("전체 대화 내용", value=get_full_transcript_txt(), height=300)
    st.download_button("⬇️ Save .txt", data=get_full_transcript_txt(),
                       file_name=file_name, mime="text/plain")
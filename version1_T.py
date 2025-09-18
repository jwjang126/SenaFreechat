# 🔊 SENA - 버튼 기반 녹음 + 대화 로그
# (LEVEL/TOPIC 선택 + 무음자동/고정길이 선택 + 전체 대화 다운로드)
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import time
import json, csv
from io import StringIO

# ===============================
# 기본 설정
# ===============================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEEP_LAST_TURNS = 8  # 모델 프롬프트에 넣을 최근 턴 수(전체 로그는 별도로 모두 보관)

# ===============================
# 모델 / TTS 로드
# ===============================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_resource
def load_tts():
    tts = TTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        progress_bar=False,
        gpu=torch.cuda.is_available()
    )
    return tts

tts = load_tts()

# ===============================
# 세션 상태
# ===============================
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # [{user: "...", ai: "..."}]

# ===============================
# 녹음 함수들
# ===============================
def record_audio_fixed(duration=5, fs=16000):
    """duration 초 동안 녹음하고 반환 (고정 길이)"""
    st.info(f"🔴 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Finished!")
    return (audio.flatten().astype(np.float32) / 32768.0), fs

def record_audio_vad(silence_sec=2.0, fs=16000, silence_thresh=0.01, max_total_sec=15):
    """무음이 silence_sec 이상 지속되면 자동 종료. 안전상 max_total_sec 넘어가면 강제 종료."""
    st.info(f"🔴 Listening… 무음 {silence_sec}초 지속 시 자동 종료 (최대 {max_total_sec}초)")
    frames = []
    block_dur = 0.05  # 50 ms
    block_size = int(fs * block_dur)
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

# ===============================
# STT
# ===============================
def run_stt(audio, fs):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    import whisper  # openai-whisper (ffmpeg 필요)
    model_w = whisper.load_model("base", device="cpu")
    result = model_w.transcribe(tmp.name, task="transcribe", language="en", temperature=0)
    return result.get("text", "").strip()

# ===============================
# LLM 응답 생성
# ===============================
def build_history_text():
    """최근 KEEP_LAST_TURNS를 'Student/Tutor' 포맷으로 직렬화"""
    lines = []
    for turn in st.session_state.conversation[-KEEP_LAST_TURNS:]:
        user = turn.get("user", "").strip()
        ai   = turn.get("ai", "").strip()
        if user:
            lines.append(f"Student: {user}")
        if ai:
            lines.append(f"Tutor: {ai}")
    return "\n".join(lines)

def generate_ai_response(user_input: str, system_prompt: str):
    conversation_text = build_history_text()
    prompt = f"""
# System
{system_prompt}

# Conversation so far
{conversation_text if conversation_text else "(no previous turns)"}

# Task
Student: "{user_input.strip()}"
Tutor:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    reply = text
    if "Tutor:" in text:
        reply = text.split("Tutor:")[-1].strip()
    if reply.startswith("Student:"):
        reply = reply.split("Student:", 1)[-1].strip()
    return reply

# ===============================
# TTS 재생
# ===============================
def speak(text):
    if not text.strip():
        text = "Hello."
    elif len(text.strip()) < 5:
        text = text.strip() + " okay."

    try:
        wav = tts.tts(text=text)
    except RuntimeError as e:
        st.error(f"TTS 오류 발생: {e}")
        text = "Let me try again."
        wav = tts.tts(text=text)

    sr = tts.synthesizer.output_sample_rate
    sd.play(wav, samplerate=sr)
    sd.wait()
    return wav, sr

# ===============================
# Streamlit UI
# ===============================
st.title("🔊 SENA - English Tutor")
st.write("🎤 버튼을 눌러 대화하세요!")

# --- (1) LEVEL / TOPIC 선택 → SYSTEM_PROMPT 반영 ---
LEVEL = st.selectbox("학습자 수준", ["beginner", "intermediate", "advanced"], index=0)
TOPIC = st.text_input("대화 주제", value="ordering food at a restaurant")

RESPONSE_LENGTH_MAP = {
    "beginner": "Answer in 1~2 short and simple sentences.",
    "intermediate": "Answer in 3~4 sentences with some detail.",
    "advanced": "Answer in 5 or more sentences with rich vocabulary and detail."
}
RESPONSE_LENGTH = RESPONSE_LENGTH_MAP[LEVEL]
SUBJECT = (
    "You are a 27-year-old American.\n"
    "Your job is an English speech teacher."
)
SYSTEM_PROMPT = f"""
The user is an English learner at {LEVEL} level.
You are their teacher, and your role is to help them improve fluency, pronunciation, and natural expressions.

Conversation topic: {TOPIC}

Guidelines:
- Always stay in character as {SUBJECT}.
- {RESPONSE_LENGTH}
- Correct the user's mistakes gently and provide the correct way to say it.
- Encourage the user to respond with slightly longer sentences each time.
- Keep your tone friendly, supportive, and engaging.
- Always keep the conversation related to the topic: "{TOPIC}".
- Ask simple follow-up questions appropriate to the user's level and the topic.
""".strip()

st.divider()

# --- (3) 녹음 종료 방식 선택 ---
mode = st.radio("발화 종료 방식", ("무음 자동 종료", "고정 길이 제한"), horizontal=True)

if mode == "무음 자동 종료":
    silence_sec = st.slider("무음 지속 시간(초)", 1.0, 5.0, 2.0, 0.5)
    silence_thresh = st.slider("무음 임계(RMS, 낮을수록 민감)", 0.002, 0.03, 0.01, 0.002)
    safety_cap = st.slider("안전 최대 길이(초)", 5, 30, 15)
    st.caption(f"💡 연속 무음이 {silence_sec}초 이상이면 자동 종료됩니다. (안전 최대 {safety_cap}초)")
else:
    duration = st.slider("발화 길이 제한 (초)", 3, 15, 5)
    st.caption(f"💡 총 발화 시간 {duration}초가 지나면 자동으로 종료됩니다.")

st.divider()

# --- 트리거 버튼 ---
if st.button("My turn"):
    # 1) 녹음
    if mode == "무음 자동 종료":
        audio, fs = record_audio_vad(
            silence_sec=silence_sec, fs=16000,
            silence_thresh=silence_thresh, max_total_sec=safety_cap
        )
    else:
        audio, fs = record_audio_fixed(duration=duration, fs=16000)

    # 2) STT
    with st.spinner("📝 인식 중..."):
        user_text = run_stt(audio, fs)

    if user_text:
        st.markdown(f"**You:** {user_text}")

        # 3) LLM
        with st.spinner("🤖 답변 생성 중..."):
            ai_reply = generate_ai_response(user_text, system_prompt=SYSTEM_PROMPT)
            st.session_state.conversation.append({"user": user_text, "ai": ai_reply})

        st.markdown(f"**AI Tutor:** {ai_reply}")

        # 4) TTS (애니메이션 없음)
        with st.spinner("🗣️ 발화 재생 중..."):
            speak(ai_reply)
    else:
        st.warning("Could not recognize any speech. Please try again.")

# --- 대화 로그(화면 표시) ---
st.markdown("## 💬 대화 내용")
for turn in st.session_state.conversation:
    if turn.get("user"):
        st.markdown(f"**You:** {turn['user']}")
    if turn.get("ai"):
        st.markdown(f"**AI Tutor:** {turn['ai']}")

# ===============================
# 📄 전체 대화 보기 / 다운로드(.md/.json/.csv)
# ===============================
def get_full_transcript_md():
    lines = []
    for turn in st.session_state.conversation:
        if turn.get("user"): lines.append(f"**You:** {turn['user']}")
        if turn.get("ai"):   lines.append(f"**AI Tutor:** {turn['ai']}")
        lines.append("---")
    return "\n\n".join(lines[:-1]) if lines else "_(no conversation yet)_"

def get_full_transcript_json():
    return json.dumps(st.session_state.conversation, ensure_ascii=False, indent=2)

def get_full_transcript_csv():
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["user", "ai"])
    writer.writeheader()
    for turn in st.session_state.conversation:
        writer.writerow({"user": turn.get("user",""), "ai": turn.get("ai","")})
    return buf.getvalue()

st.markdown("## 📄 전체 대화 보기 / 저장")
with st.expander("전체 대화 펼치기"):
    st.markdown(get_full_transcript_md())

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Save .md", data=get_full_transcript_md(),
                       file_name="conversation.md", mime="text/markdown")
with c2:
    st.download_button("⬇️ Save .json", data=get_full_transcript_json(),
                       file_name="conversation.json", mime="application/json")
with c3:
    st.download_button("⬇️ Save .csv", data=get_full_transcript_csv(),
                       file_name="conversation.csv", mime="text/csv")

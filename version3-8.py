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

# TTS 로드
@st.cache_resource
def load_tts(force_reload=False):
    if force_reload:
        st.cache_data.clear()  # streamlit cache 강제 초기화
    tts = TTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        progress_bar=False,
        gpu=torch.cuda.is_available()
    )
    return tts

tts = load_tts(force_reload=True)

# -------------------------------
# 세션 상태 초기화
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

MAX_CONTEXT = 10

# -------------------------------
# 녹음 함수
# -------------------------------
def record_audio_fixed(duration=5, fs=16000):
    st.info(f"🔴 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Finished!")
    return (audio.flatten().astype(np.float32)/32768.0), fs

def record_audio_vad(silence_sec=2.0, fs=16000, silence_thresh=0.01, max_total_sec=15):
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
            if silent_run >= silence_sec or (time.time() - start) >= max_total_sec:
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
def generate_ai_response(user_input, topic, level):
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

    prompt = f"""
## Persona
You are 'Tutor', an AI English conversation coach. Your personality is warm, endlessly curious, and patient. Your primary goal is to make the student feel comfortable and encourage them to share their own thoughts and stories, much like a friendly chat.

## Core Conversation Rules
1.  **Listen and Engage:** Always start by briefly acknowledging what the student just said.
2.  **Ask Personal, Open-ended Questions:** Your main tool is asking simple, open-ended questions about the student's personal experiences or opinions on the topic: **{topic}**.
3.  **Answering Personal Questions:** If the student asks about your preferences (e.g., "Do you like chicken?"), give a simple, natural answer (e.g., "Oh, I love it!"). Then, immediately ask the student a question to pivot back to them. Never say "I can help you learn".
4.  **Adapt to the Student's Level:** You must follow this specific instruction: **{level_instruction}**
5.  **BE A FULL SENTENCE:** Your response must be a complete sentence. Avoid one-word answers like "Yes?" or "Okay?".

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
    if not text.strip():
        st.warning("⚠️ 발화할 내용이 없습니다.")
        return

    # Tacotron2 최소 길이 방어
    min_len = 10  # 최소 토큰 길이
    if len(text) < min_len:
        text = text + " " * (min_len - len(text))  # 공백으로 길이 확보

    try:
        wav = tts.tts(text=text)
        sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()
    except RuntimeError as e:
        st.error(f"⚠️ TTS 오류 발생: {e}")

# -------------------------------
# 리포트 생성
# -------------------------------
def generate_cumulative_report(conversation_history):
    user_sentences = [turn["user"] for turn in conversation_history if "user" in turn]
    if not user_sentences:
        return "아직 대화 내용이 없습니다."

    formatted_user_sentences = "\n".join([f"{i+1}. \"{sentence}\"" for i, sentence in enumerate(user_sentences)])

    report_prompt = f"""
**당신의 역할:**
당신은 학생의 영어 문장에서 **'문법 오류'**를 찾아 **한국어로** 설명해주는 AI 영어 문법 선생님입니다.

**당신의 임무:**
아래 "학생의 문장들"을 검토하여, 문법적으로 완벽하고 자연스러운 '모범 문장'으로 교정하고, 어떤 문법 규칙이 적용되었는지 설명하는 것입니다.

**CRITICAL RULES (반드시 지켜야 할 규칙):**
1.  **학생의 이름을 부르지 마세요.**
2.  **아래 '출력 형식'과 '작성 예시'를 글자 그대로 정확하게 따르세요.**
3.  **모든 설명과 팁은 반드시 한국어로 작성하세요.**

**출력 형식:**
* **1번 문장:**
    * **모범 문장:** [교정된 영어 문장]
    * **문법 팁:** [어떤 문법 규칙 때문에 수정되었는지 한국어로 명확하게 설명]

**--- 작성 예시 ---**
* **1번 문장:**
    * **학생 발화:** "He go to school."
    * **모범 문장:** "He goes to school."
    * **문법 팁:** 주어가 3인칭 단수(He)일 때, 현재 시제 동사에는 '-s'나 '-es'를 붙여야 합니다. (주어-동사 수일치 오류)
**--- 예시 끝 ---**


**--- 학생의 문장들 ---**
{formatted_user_sentences}
**--- 문장 끝 ---**

**피드백 리포트:**
"""
    report_inputs = tokenizer(report_prompt, return_tensors="pt").to(DEVICE)
    report_outputs = model.generate(
        **report_inputs,
        max_new_tokens=500,
        temperature=0.4,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    full_report = tokenizer.decode(report_outputs[0], skip_special_tokens=True)
    if "피드백 리포트:" in full_report:
        report = full_report.split("피드백 리포트:")[-1].strip()
    else:
        report = "리포트를 생성하는 데 실패했습니다. 다시 시도해주세요."
        
    return report

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔊 SENA - English Tutor")
st.write("🎤 버튼을 눌러 대화하세요!")
topic = st.text_input("주제를 입력하세요")
level = st.selectbox("수준을 선택하세요", ["초급", "중급", "고급"])
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
# 녹음 + STT + LLM + 피드백 + TTS
# -------------------------------
if st.button("My turn"):
    if mode == "무음 자동 종료":
        audio, fs = record_audio_vad(silence_sec=silence_sec, fs=16000, silence_thresh=silence_thresh, max_total_sec=safety_cap)
    else:
        audio, fs = record_audio_fixed(duration, fs=16000)

    with st.spinner("📝 인식 중..."):
        user_text = run_stt(audio, fs)

    if user_text:
        st.markdown(f"**You:** {user_text}")
        with st.spinner("🤖 답변 생성 중..."):
            ai_reply = generate_ai_response(user_text, topic, level)
        
        st.session_state.conversation.append({
            "user": user_text,
            "ai": ai_reply,
        })
        with st.spinner("🗣️ 발화 재생 중..."):
            speak(ai_reply)
    else:
        st.warning("Could not recognize any speech. Please try again.")
st.divider()

# -------------------------------
# 📄 전체 대화 보기 / 다운로드(.txt)
# -------------------------------
def get_full_transcript_txt():
    lines = []
    for turn in st.session_state.conversation:
        if turn.get("user"): lines.append(f"You: {turn['user']}")
        if turn.get("ai"):   lines.append(f"AI Tutor: {turn['ai']}")
        lines.append("-"*40)
    return "\n".join(lines[:-1]) if lines else "(no conversation yet)"

with st.expander("📄 전체 대화 보기 / 다운로드"):
    st.text_area("전체 대화 내용", value=get_full_transcript_txt(), height=300)
    st.download_button("⬇️ Save Conversation .txt", data=get_full_transcript_txt(),
                       file_name=f"{datetime.datetime.now().strftime('%y%m%d')}_{topic}_conversation.txt", mime="text/plain")

with st.expander("📑 최종 피드백 리포트"):
    if st.button("리포트 생성"):
        with st.spinner("📝 최종 리포트를 생성합니다."):
            report = generate_cumulative_report(st.session_state.conversation)
            st.session_state.final_report = report 
    
    if st.session_state.final_report:
        st.markdown(st.session_state.final_report)
        st.download_button(
            label="⬇️ Save Report .txt",
            data=st.session_state.final_report,
            file_name=f"{datetime.datetime.now().strftime('%y%m%d')}_{topic}_FINAL_REPORT.txt",
            mime="text/plain"
        )
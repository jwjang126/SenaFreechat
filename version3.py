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
# LLM 답변 생성
# -------------------------------
def generate_ai_response(user_input):
    recent_convos = st.session_state.conversation[-MAX_CONTEXT:]
    context_text = ""
    if st.session_state.conversation_summary:
        context_text += f"[Summary of previous conversation]\n{st.session_state.conversation_summary}\n\n"
    for turn in recent_convos:
        context_text += f"Student: {turn['user']}\nTutor: {turn.get('ai','')}\n"

    prompt = f"""
You are a friendly English conversation tutor and partner.
Your Persona:
- You are a supportive and patient tutor who talks with the student like a conversation partner.
- Your goal is to make the student feel comfortable and encourage them to speak more.
- You only respond to the student’s messages.
- Keep your responses concise and to the point, while still encouraging conversation.


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

Remember: Your job is not to lecture but to **actively join the conversation** and help the student speak more.

---
**Examples of what to do and what NOT to do:**

**(O) Good Example - Asks a question:**
Me: "I enjoyed watching a movie yesterday."
You: "Wow, that sounds like so much fun! What genre of movie did you watch?
What part did you enjoy the most? Did any scene really surprise or make you laugh?"

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

Tutor:
"""

    # 안전한 토크나이저 설정
    tokenizer.pad_token = tokenizer.eos_token

    # 입력 토큰화 (길이 제한)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    # 답변 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Tutor:" in reply:
        reply = reply.split("Tutor:")[-1].strip()

    # 요약 갱신 (선택)
    if (len(st.session_state.conversation) + 1) % MAX_CONTEXT == 0:
        summary_prompt = f"Summarize the following conversation briefly:\n{context_text}"
        summary_inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        summary_outputs = model.generate(**summary_inputs, max_new_tokens=100)
        st.session_state.conversation_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

    return reply


# -------------------------------
# 피드백 생성
# -------------------------------
def generate_feedback_for_turn(user_text):
    report_prompt = f"""
You are an English tutor. Write a Feedback Report for the student's input: "{user_text}".
Focus only on what needs improvement. Do NOT mention what was done well.
Do NOT continue the conversation or ask questions. Output text only. Avoid just giving the corrected sentence without explanation.

Instructions:

1. Grammar errors: list mistakes, corrections, and explanations.
2. Awkward or unnatural expressions: suggest better alternatives and explain why.
3. Over-translation or excessive literal translation: note if found and explain.
4. Vocabulary suggestions: for advanced students, recommend better words/phrases to sound more native-like.
5. Score: give a score out of 10 (10 = native-like, 1 = very poor). Focus on grammar, clarity, naturalness, and vocabulary.
6. Summary: 2–3 sentences highlighting improvements needed. For 초급, prioritize must-fix errors. For 중급, include moderate corrections. For 고급, prioritize advanced vocabulary and natural phrasing.

Level guidance:
- 초급: Focus on essential corrections, prioritize grammar and clarity.
- 중급: Include grammar and stylistic improvements, suggest better words/phrases moderately.
- 고급: Be thorough and detailed, suggest advanced vocabulary, idioms, and natural expressions.

Examples (TODO / NOT TODO):

**(O) Good Example - Correct usage (ignore in report)**:
Me: "I went to the park yesterday."

(O) Good Example – Recommend a better response with an appropriate explanation of the grammatical errors:
Student: "I go to park yesterday."
Tutor: "I went to the park yesterday." Since you are talking about something that happened in the past (yesterday), the verb should be in the past tense. 'go' becomes 'went'. Also, 'park' needs the article 'the' to be grammatically correct.

(X) Bad Example – Simply provide the answer:
Student: "I go to park yesterday."
Tutor: "I went to the park yesterday."

(O) Good Example – Suggest a better sentence with an appropriate explanation for the unnatural phrasing:
Student: "I like read book."
Tutor: "I like reading books." After 'like', we usually use the gerund form of the verb, so 'read' becomes 'reading'. Also, when talking about books in general, use the plural 'books' to make it sound natural.

(O) Good Example – Combine grammar + unnatural phrasing corrections:
Student: "She no go to school today."
Tutor: "She didn’t go to school today." The verb 'go' must be in the past tense because it refers to today’s absence. Also, use 'didn’t' for negative past tense sentences.

(X) Bad Example – Only give a corrected sentence without explanation:
Student: "She no go to school today."
Tutor: "She didn’t go to school today."


##Send only the answer of "Tutor" in print.
---------------------------------
Report format:

Grammar errors:
- mistake → correction + explanation

Awkward or unnatural expressions:
- original → improved + explanation

Over-translation:
- note if applicable + explanation

Vocabulary suggestions (if applicable):
- original → suggestion + explanation

Score: X/10

Summary:
- ...

"""
    report_inputs = tokenizer(report_prompt, return_tensors="pt").to(DEVICE)
    report_outputs = model.generate(**report_inputs, max_new_tokens=200, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
    report = tokenizer.decode(report_outputs[0], skip_special_tokens=True)
    return report

# -------------------------------
# TTS
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
sentence_count = {"초급": 2, "중급": 4, "고급": 6}[level]

st.sidebar.header("🎚️ 녹음 설정")
mode = st.sidebar.radio("발화 종료 방식", ("무음 자동 종료", "고정 길이 제한"), horizontal=True)
if mode == "무음 자동 종료":
    silence_sec = st.sidebar.slider("무음 지속 시간(초)", 1.0, 5.0, 2.0, 0.5)
    with st.expander("고급 설정 보기"):
        silence_thresh = st.sidebar.slider("무음 임계(RMS x100)", 0.2, 3.0, 1.0, 0.4)/100
        safety_cap = st.sidebar.slider("최대 발화 시간(초)", 5, 30, 15)
else:
    duration = st.sidebar.slider("발화 길이 제한 (초)", 3, 15, 5)

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
            ai_reply = generate_ai_response(user_text)
        with st.spinner("📝 피드백 생성 중..."):
            report = generate_feedback_for_turn(user_text)
        # 세션에 모두 저장
        st.session_state.conversation.append({
            "user": user_text,
            "ai": ai_reply,
            "report": report
        })
        with st.spinner("🗣️ 발화 재생 중..."):
            speak(ai_reply)
    else:
        st.warning("Could not recognize any speech. Please try again.")

# -------------------------------
# 전체 대화 + 피드백 보기 / 다운로드
# -------------------------------
def get_full_transcript_txt():
    lines = []
    for turn in st.session_state.conversation:
        if turn.get("user"): lines.append(f"You: {turn['user']}")
        if turn.get("ai"): lines.append(f"Tutor: {turn['ai']}")
        lines.append("-"*40)
    return "\n".join(lines[:-1]) if lines else "(no conversation yet)"

def get_full_report_txt():
    lines = []
    for i, turn in enumerate(st.session_state.conversation, 1):
        if turn.get("user"): lines.append(f"[Turn {i}] Student: {turn['user']}")
        if turn.get("ai"): lines.append(f"Tutor: {turn['ai']}")
        if turn.get("report"): lines.append(f"Feedback Report:\n{turn['report']}")
        lines.append("="*50)
    return "\n".join(lines)

today_str = datetime.datetime.now().strftime("%y%m%d")
safe_topic = topic.replace(" ", "_") if topic else "NoTopic"

with st.expander("📄 전체 대화 보기 / 다운로드"):
    st.text_area("전체 대화 내용", value=get_full_transcript_txt(), height=300)
    st.download_button("⬇️ Save .txt", data=get_full_transcript_txt(),
                       file_name=f"{today_str}_{safe_topic}_{level}.txt", mime="text/plain")

with st.expander("📑 발화 평가 리포트 보기 / 다운로드"):
    st.text_area("전체 리포트 내용", value=get_full_report_txt(), height=300)
    st.download_button("⬇️ Save Report .txt", data=get_full_report_txt(),
                       file_name=f"{today_str}_{safe_topic}_{level}_REPORT.txt", mime="text/plain")

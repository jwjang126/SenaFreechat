# 🔊 SENA - 텍스트 기반 대화 + 대화 로그
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import sounddevice as sd
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
7.  When the user says something, respond with empathy. Focus on casual, everyday conversation rather than providing factual information.

The student has provided the following topic: "{user_input}".

Level guidance:
- Difficulty: {level}
- Student description:
    - 초급: Think the student is a high school student. Use simple words and short sentences.
    - 중급: Think the student is an adult learner. Use moderate vocabulary and clear explanations.
    - 고급: Think the student wants to speak like a native speaker. Use advanced vocabulary and natural expressions.
- Expected response length: around {sentence_count} sentences.

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

    # Tutor: 뒤에 오는 부분만 추출
    if "Tutor:" in reply:
        reply = reply.split("Tutor:")[-1].strip()

    if (len(st.session_state.conversation) + 1) % MAX_CONTEXT == 0:
        summary_prompt = f"Summarize the following conversation briefly:\n{context_text}"
        summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to(DEVICE)
        summary_outputs = model.generate(**summary_inputs, max_new_tokens=100)
        st.session_state.conversation_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

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
st.write("✏️ 텍스트로 대화해 보세요!")
topic = st.text_input("주제를 입력하세요")
level = st.selectbox("수준을 선택하세요", ["초급", "중급", "고급"])

# 수준별 문장 수 설정
sentence_count = {"초급": 2, "중급": 4, "고급": 6}[level]

st.divider()

# -------------------------------
# 텍스트 입력 + LLM + TTS 트리거
# -------------------------------
user_text = st.text_input("💬 Your message:")

if st.button("Send"):
    if user_text.strip():
        st.markdown(f"**You:** {user_text}")
        with st.spinner("🤖 답변 생성 중..."):
            ai_reply = generate_ai_response(user_text)
            st.session_state.conversation.append({"user": user_text, "ai": ai_reply})
        with st.spinner("🗣️ 발화 재생 중..."):
            speak(ai_reply)
    else:
        st.warning("메시지를 입력해주세요.")

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

today_str = datetime.datetime.now().strftime("%y%m%d")
safe_topic = topic.replace(" ", "_") if topic else "NoTopic"
file_name = f"{today_str}_{safe_topic}_{level}.txt"

with st.expander("📄 전체 대화 보기 / 다운로드"):
    st.text_area("전체 대화 내용", value=get_full_transcript_txt(), height=300)
    st.download_button("⬇️ Save .txt", data=get_full_transcript_txt(),
                       file_name=file_name, mime="text/plain")

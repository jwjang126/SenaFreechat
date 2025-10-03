# backend/process_audio.py
import sys
import os
import json
import subprocess
import whisper
from TTS.api import TTS
import logging
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s:%(name)s:%(message)s')

# --- 모델 로딩 부분 (수정 없음) ---
pipe = None
whisper_model = None
tts_model = None
try:
    print("Loading AI models... This might take a moment.", file=sys.stderr)
    
    # 1. LLM 모델 로드 (torch_dtype -> dtype으로 수정 권장)
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map="auto")
    pipe.tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
        "{% endfor %}"
        "{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
    )
    
    # 2. STT (Whisper) 모델 로드
    whisper_model = whisper.load_model("base")

    # 3. TTS 모델 로드
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    
    print("All models loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"Failed to load models: {e}", file=sys.stderr)
# -----------------------------------------------------------

def main():
    # --- 수정된 부분 1: 인자 처리 ---
    if len(sys.argv) < 6:
        print(json.dumps({"error":"need args: input_wav mp3_out conversation_history_json topic level"}))
        return

    input_wav = sys.argv[1]
    mp3_out = sys.argv[2]
    conversation_history = json.loads(sys.argv[3]) if sys.argv[3] else []
    topic = sys.argv[4]
    level = sys.argv[5]
    # --------------------------------

    try:
        if pipe is None or whisper_model is None or tts_model is None:
            raise RuntimeError("AI models are not loaded. Cannot proceed.")

        # 1. STT (Whisper) - 미리 로드된 모델 사용
        result = whisper_model.transcribe(input_wav)
        user_text = result.get("text", "").strip()

        # 2. LLM 호출
        print(f"User said: {user_text}", file=sys.stderr)
        
        # --- 수정된 부분 2: 동적 프롬프트 생성 ---
        system_prompt_content = f"""You are 'Tutor', a friendly and engaging English conversation partner.
Today's conversation topic is: {topic}.
Please adjust your language difficulty to a {level} level for the student.

**Core Instructions:**
- Act like a curious and encouraging friend, not a robot. Be natural.
- Ask about the student's personal experiences, memories, or preferences related to the topic.
- Avoid overly abstract or philosophical questions.
- Keep your replies relatively short and easy to understand.
- ALWAYS end your response with an open-ended question to keep the conversation flowing.

**Crucial Restrictions:**
- NEVER break character. Do not output meta-commentary or instructions.
- You MUST NEVER repeat the student's sentences or key phrases. Paraphrase instead.
- Your final response MUST end with a punctuation mark like '.', '?', or '!'.

EXAMPLE:
User: "I like listening music."
Assistant: "Music is a wonderful hobby! What's your favorite genre?"
"""
        system_prompt = {
            "role": "system",
            "content": system_prompt_content
        }
        # ---------------------------------------------
        
        messages = [system_prompt] + conversation_history + [{"role": "user", "content": user_text}]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        ai_reply = outputs[0]["generated_text"][len(prompt):].strip()
        print(f"AI replied: {ai_reply}", file=sys.stderr)

        # 3. TTS - 미리 로드된 모델 사용
        temp_wav = os.path.join(os.path.dirname(input_wav), "temp_tts.wav")
        tts_model.tts_to_file(text=ai_reply, file_path=temp_wav)

        # 4. MP3 변환
        subprocess.run(["ffmpeg", "-y", "-i", temp_wav, mp3_out], check=True, capture_output=True, text=True)

        try: os.remove(temp_wav)
        except: pass

        output = {"user_text": user_text, "ai_reply": ai_reply}
        print(json.dumps(output))

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(json.dumps({"error": error_message}))
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
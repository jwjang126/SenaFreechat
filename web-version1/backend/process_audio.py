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

# 모델 로딩 부분은 동일
pipe = None
try:
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
    pipe.tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
        "{% endfor %}"
        "{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
    )
    print("Hugging Face model loaded.", file=sys.stderr)
except Exception as e:
    pipe = None
    print(f"Failed to load Hugging Face model: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error":"need args: input_wav mp3_out [conversation_history_json]"}))
        return

    input_wav = sys.argv[1]
    mp3_out = sys.argv[2]
    conversation_history = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []

    try:
        if pipe is None:
            raise RuntimeError("LLM model is not loaded.")

        # 1. STT (Whisper)
        model = whisper.load_model("base")
        result = model.transcribe(input_wav)
        user_text = result.get("text", "").strip()

        # 2. LLM 호출
        print(f"User said: {user_text}", file=sys.stderr)
        
        # [수정] 프롬프트 강화: 문장 부호 규칙 추가
        system_prompt = {
        "role": "system",
        "content": """You are a friendly English conversation tutor. 
        Your Persona:- You are friendly, encouraging, and patient.
                    - Your goal is to make the student feel comfortable and encourage them to speak more.

        RULES:
        1.  Continue the conversation naturally.
        2.  **Paraphrase, Don't Repeat:** You MUST NEVER repeat the student's sentences or key phrases.
        3.  CRITICAL RULE: Your final response MUST end with a punctuation mark like '.', '?', or '!'.

        EXAMPLE:
        User: "I like listening music."
        Assistant: "Music is a wonderful hobby! What's your favorite genre, and why do you like it?"
"""
    }
        
        messages = [system_prompt] + conversation_history + [{"role": "user", "content": user_text}]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        ai_reply = outputs[0]["generated_text"][len(prompt):].strip()
        print(f"AI replied: {ai_reply}", file=sys.stderr)

        # 3. TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        temp_wav = os.path.join(os.path.dirname(input_wav), "temp_tts.wav")
        tts.tts_to_file(text=ai_reply, file_path=temp_wav)

        subprocess.run(["ffmpeg", "-y", "-i", temp_wav, mp3_out], check=True, capture_output=True, text=True)

        try: os.remove(temp_wav)
        except: pass

        # [수정] 최종 결과물에 user_text를 추가하여 반환
        output = {"user_text": user_text, "ai_reply": ai_reply}
        print(json.dumps(output))

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(json.dumps({"error": error_message}))
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
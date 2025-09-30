// src/pages/Freechat.js
import { useState, useRef } from "react";

function Freechat() {
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("초급");
  const [conversation, setConversation] = useState([]);
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    if (audioURL) {
      try {
        const filename = audioURL.split("/").pop();
        await fetch("http://localhost:5000/delete-audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: filename }),
        });
      } catch (err) {
        console.error("Failed to request audio deletion:", err);
      }
      setAudioURL(null);
    }

    setRecording(true);
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: "audio/webm;codecs=opus" };
      mediaRecorderRef.current = new MediaRecorder(stream, options);

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm;codecs=opus" });
        if (blob.size === 0) {
          alert("녹음된 내용이 없습니다.");
          return;
        }

        const formData = new FormData();
        formData.append("audio", blob, "record.webm");
        formData.append('conversationHistory', JSON.stringify(conversation));

        try {
          const response = await fetch("http://localhost:5000/process-audio", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) throw new Error("server error " + response.status);

          const data = await response.json();
          
          // [수정] user와 ai의 대화를 conversation 상태에 함께 저장
          setConversation((prev) => [
            ...prev,
            { user: data.user_text, ai: data.ai_reply }
          ]);

          if (data.audio_file) {
            const url = `http://localhost:5000/${data.audio_file}`;
            setAudioURL(url);
            const audio = new Audio(url);
            // 자동재생 시도 (브라우저 정책에 따라 실패할 수 있음)
            audio.play().catch(e => console.log("Autoplay was prevented.", e));
          }
        } catch (err) {
          console.error("오디오 처리 실패:", err);
        }
      };

      mediaRecorderRef.current.start();
    } catch (err) {
      console.error("녹음 실패:", err);
      setRecording(false);
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  // JSX (화면 UI) 부분은 그대로입니다.
  return (
    <div style={{ padding: "1rem" }}>
      <h2>🎤 AI TUTOR</h2>
      <input
        type="text"
        placeholder="주제를 입력하세요"
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        style={{ marginRight: "1rem" }}
      />
      <select value={level} onChange={(e) => setLevel(e.target.value)}>
        <option>초급</option>
        <option>중급</option>
        <option>고급</option>
      </select>
      <br /><br />
      {!recording ? (
        <button onClick={startRecording}>녹음 시작</button>
      ) : (
        <button onClick={handleStopRecording}>녹음 종료</button>
      )}
      <br /><br />
      {audioURL && (
        <div>
          <p>AI 음성 재생:</p>
          <audio src={audioURL} controls />
        </div>
      )}
      <br />
      <div style={{ border: "1px solid #ccc", padding: "10px", height: "200px", overflowY: "scroll" }}>
        {conversation.map((turn, index) => (
          <div key={index}>
            <b>You:</b> {turn.user} <br />
            <b>AI Tutor:</b> {turn.ai}
            <hr />
          </div>
        ))}
      </div>
    </div>
  );
}

export default Freechat;
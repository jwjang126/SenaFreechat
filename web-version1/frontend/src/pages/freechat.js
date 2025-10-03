import { useState, useRef } from "react";

// 무음 감지를 위한 커스텀 훅 (컴포넌트 외부에 정의)
function useVAD(onStop, silenceDurationMs = 2000) {
  const audioContextRef = useRef(null);
  const mediaStreamSourceRef = useRef(null);
  const scriptProcessorNodeRef = useRef(null);
  const silenceStartRef = useRef(Date.now());

  const start = (stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = audioContext;
    const mediaStreamSource = audioContext.createMediaStreamSource(stream);
    mediaStreamSourceRef.current = mediaStreamSource;
    const bufferSize = 4096;
    const scriptProcessorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
    scriptProcessorNodeRef.current = scriptProcessorNode;
    const SILENCE_THRESHOLD = 0.01;

    scriptProcessorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      let sum = 0.0;
      for (let i = 0; i < input.length; ++i) {
        sum += input[i] * input[i];
      }
      const rms = Math.sqrt(sum / input.length);

      if (rms > SILENCE_THRESHOLD) {
        silenceStartRef.current = Date.now();
      } else {
        if (Date.now() - silenceStartRef.current > silenceDurationMs) {
          onStop();
          stop();
        }
      }
    };
    mediaStreamSource.connect(scriptProcessorNode);
    scriptProcessorNode.connect(audioContext.destination);
    silenceStartRef.current = Date.now();
  };

  const stop = () => {
    if (mediaStreamSourceRef.current) mediaStreamSourceRef.current.disconnect();
    if (scriptProcessorNodeRef.current) scriptProcessorNodeRef.current.disconnect();
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
  };

  return { start, stop };
}

// 메인 Freechat 컴포넌트
function Freechat() {
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("초급");
  const [conversation, setConversation] = useState([]);
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [silenceDuration, setSilenceDuration] = useState(2);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const vad = useVAD(handleStopRecording, silenceDuration * 1000);

  const startRecording = async () => {
    if (audioURL) {
      try {
        const filename = audioURL.split("/").pop();
        await fetch("http://localhost:5000/delete-audio", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename }),
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
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        vad.stop();
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm;codecs=opus" });
        if (blob.size === 0) {
          alert("녹음된 내용이 없습니다.");
          setRecording(false);
          return;
        }

        const formData = new FormData();
        formData.append("audio", blob, "record.webm");
        const historyForAI = conversation.flatMap(turn => [
        { role: 'user', content: turn.user },
        { role: 'assistant', content: turn.ai }
      ]);
      formData.append('conversationHistory', JSON.stringify(historyForAI));
      formData.append('level', level);
      formData.append('topic', topic); 

        try {
          const response = await fetch("http://localhost:5000/process-audio", { method: "POST", body: formData });
          if (!response.ok) throw new Error("server error " + response.status);
          const data = await response.json();
          setConversation((prev) => [...prev, { user: data.user_text, ai: data.ai_reply }]);
          if (data.audio_file) {
            const url = `http://localhost:5000/${data.audio_file}`;
            setAudioURL(url);
            const audio = new Audio(url);
            audio.play().catch(e => console.log("Autoplay was prevented.", e));
          }
        } catch (err) {
          console.error("오디오 처리 실패:", err);
        }
      };

      mediaRecorderRef.current.start();
      vad.start(stream);
    } catch (err) {
      console.error("녹음 실패:", err);
      setRecording(false);
    }
  };

  const manualStopRecording = () => {
    vad.stop();
    handleStopRecording();
  };

  return (
    <div style={{ display: "flex", fontFamily: "sans-serif" }}>
      <div style={{
        width: isSidebarOpen ? "250px" : "0px",
        padding: isSidebarOpen ? "1rem" : "0",
        borderRight: isSidebarOpen ? "1px solid #eee" : "none",
        background: "#f9f9f9",
        overflow: "hidden",
        transition: "width 0.3s ease, padding 0.3s ease"
      }}>
        <h3>🎚️ 녹음 설정</h3>
        <hr/>
        <label htmlFor="silence-duration" style={{ fontSize: '0.9rem', color: '#333' }}>무음 지속 시간 (초)</label>
        <p style={{ textAlign: "center", fontWeight: "bold", fontSize: '1.2rem', color: '#007bff', margin: '0.5rem 0' }}>{silenceDuration.toFixed(1)}초</p>
        <input
          type="range"
          id="silence-duration"
          min="1"
          max="5"
          step="0.5"
          value={silenceDuration}
          onChange={(e) => setSilenceDuration(parseFloat(e.target.value))}
          style={{ width: "100%" }}
        />
        <hr/>
        <p style={{ fontSize: '0.8rem', color: '#666' }}>
          💡 녹음 시작 후, 설정된 시간 동안 말이 없으면 자동으로 녹음이 종료됩니다.
        </p>
      </div>

      <div style={{ flex: 1, padding: "1rem 2rem", position: 'relative' }}>
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          style={{
            position: 'absolute',
            left: isSidebarOpen ? '-15px' : '15px',
            top: '1rem',
            background: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '30px',
            height: '30px',
            cursor: 'pointer',
            fontSize: '1rem',
            lineHeight: '30px',
            textAlign: 'center',
            transition: 'left 0.3s ease',
            zIndex: 10
          }}
        >
          {isSidebarOpen ? "‹" : "›"}
        </button>

        <h2>🎤 AI TUTOR</h2>
        
        <div style={{ marginBottom: "1.5rem" }}>
          <input
              type="text"
              placeholder="주제를 입력하세요"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              style={{ marginRight: "1rem", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
          />
          <select value={level} onChange={(e) => setLevel(e.target.value)} style={{ padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}>
              <option>초급</option>
              <option>중급</option>
              <option>고급</option>
          </select>
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          {!recording ? (
              <button onClick={startRecording} style={{ padding: '10px 20px', fontSize: '1rem', cursor: 'pointer' }}>녹음 시작</button>
          ) : (
              <button onClick={manualStopRecording} style={{ padding: '10px 20px', fontSize: '1rem', cursor: 'pointer', background: '#dc3545', color: 'white' }}>녹음 수동 종료</button>
          )}
        </div>
        
        {audioURL && (
            <div style={{ marginBottom: "1.5rem" }}>
                <p>AI 음성 재생:</p>
                <audio src={audioURL} controls />
            </div>
        )}

        <div style={{ border: "1px solid #eee", background: "#fff", borderRadius: "8px", padding: "10px", height: "300px", overflowY: "scroll" }}>
            {conversation.length === 0 ? (
                <p style={{ color: '#888' }}>(대화 기록 없음)</p>
            ) : (
                conversation.map((turn, index) => (
                    <div key={index} style={{ marginBottom: '1rem' }}>
                        <p style={{ margin: '0 0 5px 0' }}><b>You:</b> {turn.user}</p>
                        <p style={{ margin: 0, color: '#007bff' }}><b>AI Tutor:</b> {turn.ai}</p>
                        {index < conversation.length - 1 && <hr style={{ border: 0, borderTop: '1px solid #eee', margin: '1rem 0' }} />}
                    </div>
                ))
            )}
        </div>
      </div>
    </div>
  );
}

export default Freechat;


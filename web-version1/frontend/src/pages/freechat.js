import { useState, useRef } from "react";

function Freechat() {
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("초급");
  const [conversation, setConversation] = useState([]);
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // 녹음 시작
  const startRecording = async () => {
    setRecording(true);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);

        // 예시: AI 답변 대신 더미 텍스트
        setConversation((prev) => [
          ...prev,
          { user: topic || "(주제 없음)", ai: "AI가 답변할 자리" },
        ]);
      };

      mediaRecorderRef.current.start();
    } catch (err) {
      console.error("녹음 실패:", err);
      setRecording(false);
    }
  };

  // 녹음 종료
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div style={{ padding: "1rem" }}>
      <h2>🎤 Freechat 화면</h2>

      {/* 주제 입력 */}
      <input
        type="text"
        placeholder="주제를 입력하세요"
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        style={{ marginRight: "1rem" }}
      />

      {/* 난이도 선택 */}
      <select value={level} onChange={(e) => setLevel(e.target.value)}>
        <option>초급</option>
        <option>중급</option>
        <option>고급</option>
      </select>

      <br /><br />

      {/* 녹음 버튼 */}
      {!recording ? (
        <button onClick={startRecording}>녹음 시작</button>
      ) : (
        <button onClick={stopRecording}>녹음 종료</button>
      )}

      <br /><br />

      {/* 녹음 재생 */}
      {audioURL && (
        <div>
          <p>녹음 재생:</p>
          <audio src={audioURL} controls />
        </div>
      )}

      <br />

      {/* 대화 내역 */}
      <div
        style={{
          border: "1px solid #ccc",
          padding: "10px",
          height: "200px",
          overflowY: "scroll",
        }}
      >
        {conversation.length === 0 ? (
          <p>(대화 기록 없음)</p>
        ) : (
          conversation.map((turn, index) => (
            <div key={index}>
              <b>You:</b> {turn.user} <br />
              <b>AI Tutor:</b> {turn.ai}
              <hr />
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Freechat;

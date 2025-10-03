// backend/server.js
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use("/media", express.static(path.join(__dirname, "uploads")));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const upload = multer({ dest: "uploads/" });

app.post("/process-audio", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  try {
    // 1. Python API 서버로 보낼 새로운 FormData 생성
    const formData = new FormData();
    formData.append("audio", fs.createReadStream(req.file.path), { filename: 'record.webm' });
    formData.append("conversationHistory", req.body.conversationHistory || "[]");
    formData.append("topic", req.body.topic || "any interesting topic");
    formData.append("level", req.body.level || "beginner");

    // 2. Python FastAPI 서버에 POST 요청
    const pythonApiUrl = "http://localhost:8000/process-audio/";
    const response = await axios.post(pythonApiUrl, formData, {
      headers: formData.getHeaders(),
    });

    const data = response.data;
    if (data.error) {
        throw new Error(data.error);
    }

    // 3. Base64로 인코딩된 오디오 데이터를 다시 mp3 파일로 저장
    const audioBuffer = Buffer.from(data.audio_base64, 'base64');
    const mp3FileName = req.file.filename + ".mp3";
    const mp3Path = path.join(__dirname, "uploads", mp3FileName);
    fs.writeFileSync(mp3Path, audioBuffer);
    
    // 4. React 클라이언트에 최종 결과 전송
    res.json({
      user_text: data.user_text,
      ai_reply: data.ai_reply,
      audio_file: "media/" + mp3FileName, // React가 접근할 수 있는 경로
    });

  } catch (error) {
    console.error("Error proxying to Python API:", error.message);
    res.status(500).json({ error: "Failed to process audio via Python API" });
  } finally {
    // 5. multer가 생성한 임시 webm 파일 삭제
    fs.unlinkSync(req.file.path);
  }
});
// 파일 삭제 API
app.post("/delete-audio", (req, res) => {
  const filename = req.body.filename;
  if (!filename || filename.includes("..")) {
    return res.status(400).json({ error: "Invalid filename" });
  }
  const filePath = path.join(__dirname, "uploads", filename);
  if (fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
      res.json({ message: "File deleted successfully" });
    } catch (err) {
      res.status(500).json({ error: "Failed to delete file" });
    }
  } else {
    res.json({ message: "File not found" });
  }
});

app.listen(5000, () => console.log("Server running on port 5000"));

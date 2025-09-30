// backend/server.js
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");

const app = express();
app.use(cors());
app.use("/media", express.static(path.join(__dirname, "uploads")));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const upload = multer({ dest: "uploads/" });

app.post("/process-audio", upload.single("audio"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const conversationHistory = req.body.conversationHistory || "[]";
  const audioPath = req.file.path;
  const pcmPath = audioPath + "_16k.wav";
  const mp3Path = path.join(__dirname, "uploads", path.basename(audioPath, ".webm") + ".mp3");

  ffmpeg(audioPath)
    .outputOptions(["-ac 1", "-ar 16000", "-f wav"])
    .output(pcmPath)
    .on("end", () => {
      const py = spawn("python", [
        path.join(__dirname, "process_audio.py"),
        pcmPath,
        mp3Path,
        conversationHistory,
      ]);

      let result = "";
      py.stdout.on("data", (chunk) => { result += chunk.toString(); });
      py.stderr.on("data", (c) => { console.error("Python stderr:", c.toString()); });
      py.on("close", (code) => {
        console.log("Python exited with code:", code);
        try {
          const lines = result.trim().split('\n');
          const lastLine = lines[lines.length - 1];
          const jsonResult = JSON.parse(lastLine);

          if (!fs.existsSync(mp3Path)) {
            return res.status(500).json({ error: "MP3 not produced" });
          }

          // [수정] 프론트엔드로 user_text와 ai_reply를 모두 전달
          res.json({
            user_text: jsonResult.user_text,
            ai_reply: jsonResult.ai_reply,
            audio_file: "media/" + path.basename(mp3Path)
          });
          
          try { fs.unlinkSync(audioPath); } catch(e){}
          try { fs.unlinkSync(pcmPath); } catch(e){}

        } catch (err) {
          console.error("Failed to parse JSON:", err);
          res.status(500).json({ error: "Python output parse error", raw: result });
        }
      });
    })
    .on("error", (err) => {
      console.error("FFmpeg error:", err);
      res.status(500).json({ error: "Audio conversion failed" });
    })
    .run();
});

// 파일 삭제 API (이전과 동일)
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
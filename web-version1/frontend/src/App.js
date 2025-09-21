import React, { useEffect, useState } from "react";
import './App.css';

function App() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch("/api/hello") // proxy 설정 후
      .then(res => res.json())
      .then(data => setMsg(data.message))
      .catch(err => setMsg("Error fetching data"));
  }, []);

  return (
    <div className="App">
      <h1>{msg}</h1>
    </div>
  );
}

export default App;

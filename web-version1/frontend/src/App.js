// App.js
import { useState } from "react";
import Chat from "./pages/freechat"; // 컴포넌트는 대문자로 시작
import Home from "./pages/home"; 

function App() {
  const [page, setPage] = useState("home");

  return (
    <div>
      <header style={{ background: "#333", color: "#fff", padding: "1rem" }}>
        <button onClick={() => setPage("home")}>Home</button>
        <button onClick={() => setPage("chat")}>Freechat</button>
        {/* <button onClick={() => setPage("about")}>소개</button> */}
      </header>

      <main style={{ padding: "2rem" }}>
        {page === "home" && <Home />}
        {page === "chat" && <Chat />}
      </main>
    </div>
  );
}

export default App;

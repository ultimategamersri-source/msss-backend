// -------------------------
// Backend API base
// -------------------------
const API = "/api";  // Netlify will forward to Cloud Run

// -------------------------
// Utility to show status or errors
// -------------------------
function showStatus(msg) {
  console.log(msg); // Hook this to your UI if you want
}

// -------------------------
// Test backend connection
// -------------------------
async function ping() {
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) throw new Error(`Health ${res.status}`);
    const data = await res.json();
    console.log("✅ Backend Health:", data);
  } catch (err) {
    console.error("❌ Backend not reachable:", err);
    showStatus("⚠️ Server not reachable.");
  }
}

// -------------------------
// Chat Endpoint
// -------------------------
async function chat(message) {
  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
    const data = await res.json();
    return data.reply;
  } catch (err) {
    console.error(err);
    showStatus("⚠️ Chat server error");
    return null;
  }
}

// -------------------------
// Ask Endpoint (LLM / School Assistant)
// -------------------------
async function ask(question) {
  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`Ask failed: ${res.status}`);
    const data = await res.json();
    return data.answer;
  } catch (err) {
    console.error(err);
    showStatus("⚠️ Assistant server error");
    return null;
  }
}

// -------------------------
// UI Logic — Chat Submit
// -------------------------
document.addEventListener("DOMContentLoaded", () => {
  ping(); // check backend when page loads

  const form = document.getElementById("chat-form");
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");

  if (!form || !input || !chatBox) {
    console.warn("Chat elements not found in DOM.");
    return;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    // Show user message in chat box
    chatBox.innerHTML += `<div class="msg user">You: ${question}</div>`;

    // Get assistant response
    const reply = await ask(question);

    // Show assistant reply
    chatBox.innerHTML += `<div class="msg bot">Brightly: ${reply || "⚠️ Error replying"}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    input.value = "";
  });
});

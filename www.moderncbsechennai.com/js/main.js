// const API = "https://msss-backend-665722959305.asia-south1.run.app";  // Cloud Run backend URL
const API = "/api";  // Netlify will proxy this to Cloud Run

// ✅ Test backend connection
fetch(`${API}/health`)
  .then(res => res.json())
  .then(data => console.log("Backend connected:", data))
  .catch(err => console.error("Backend error:", err));

// ✅ Example POST request to /chat
function sendChat(message) {
  return fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  }).then(res => res.json());
}

// ✅ Example POST request to /ask
function askQuestion(question) {
  return fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  }).then(res => res.json());
}

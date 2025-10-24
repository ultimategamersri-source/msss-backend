//const API = "https://msss-backend-665722959305.asia-south1.run.app";  // Cloud Run backend URL
const API = "/api";
// Example test call
fetch(`${API}/chat`, {...})
fetch(`${API}/ask`, {...})
fetch(`${API}/health`)
  .then(res => res.json())
  .then(data => console.log("Backend connected:", data))
  .catch(err => console.error("Backend error:", err));

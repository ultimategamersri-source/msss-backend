// js/chat.js

// ---- Guards ---------------------------------------------------------------
if (typeof API_BASE === "undefined") {
  console.error("[chat.js] API_BASE is not defined. Make sure js/config.js is loaded BEFORE js/chat.js.");
}

// ---- DOM refs -------------------------------------------------------------
const chatBtn    = document.getElementById('chat-btn');
const chatWindow = document.getElementById('chat-window');
const sendBtn    = document.getElementById('send-btn');
const userInput  = document.getElementById('user-input');
const chatBody   = document.getElementById('chat-body');
const header     = document.getElementById('chat-header');

// ---- State ----------------------------------------------------------------
let waiting = false;
let chatOpen = false;
let chatHistory = [{ type: 'bot', text: 'Hello! Ask me about school.' }];

// ---- UI helpers -----------------------------------------------------------
function renderMessages(){
  chatBody.innerHTML = '';
  chatHistory.forEach(msg => {
    const bubble = document.createElement('div');
    bubble.className = `message ${msg.type}`;
    if (msg.type === 'bot') {
      const icon = document.createElement('span');
      icon.className = 'icon';
      icon.textContent = '💡';
      bubble.appendChild(icon);
    }
    bubble.appendChild(document.createTextNode(msg.text));
    chatBody.appendChild(bubble);
  });
  chatBody.scrollTop = chatBody.scrollHeight;
}

function autoResizeTextarea(){
  if (!userInput) return;
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
}

// ---- Draggable window -----------------------------------------------------
let isDragging = false, offsetX = 0, offsetY = 0;
if (header && chatWindow) {
  header.addEventListener('mousedown', e => {
    isDragging = true;
    offsetX = e.clientX - chatWindow.offsetLeft;
    offsetY = e.clientY - chatWindow.offsetTop;
    header.style.cursor = 'grabbing';
  });
  document.addEventListener('mousemove', e => {
    if (isDragging) {
      chatWindow.style.left = (e.clientX - offsetX) + 'px';
      chatWindow.style.top  = (e.clientY - offsetY) + 'px';
    }
  });
  document.addEventListener('mouseup', () => {
    isDragging = false;
    header.style.cursor = 'grab';
  });
}

// ---- Toggle open/close ----------------------------------------------------
if (chatBtn && chatWindow) {
  chatBtn.addEventListener('click', () => {
    chatOpen = !chatOpen;
    chatWindow.style.display = chatOpen ? 'flex' : 'none';
    autoResizeTextarea();
  });
}

// ---- API helper -----------------------------------------------------------
async function postJSON(url, bodyObj) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj)
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`${url} -> ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

// ---- Send flow ------------------------------------------------------------
async function sendMessage(){
  if (waiting) return;
  const text = (userInput?.value || '').trim();
  if (!text) return;

  chatHistory.push({ type: 'user', text });
  renderMessages();
  if (userInput) {
    userInput.value = '';
    autoResizeTextarea();
  }

  chatHistory.push({ type: 'bot', text: '💭 Thinking...' });
  renderMessages();
  waiting = true;

  try {
    const data = await postJSON(`${API_BASE}/ask`, { question: text });
    chatHistory.pop(); // remove "Thinking..."
    chatHistory.push({ type: 'bot', text: data?.answer || "I couldn't find an answer." });
    renderMessages();
  } catch (e) {
    console.error(e);
    chatHistory.pop();
    chatHistory.push({ type: 'bot', text: '⚠️ Server not reachable. Please try again later.' });
    renderMessages();
  } finally {
    waiting = false;
  }
}

// ---- Wire events ----------------------------------------------------------
if (sendBtn) {
  sendBtn.addEventListener('click', sendMessage);
}
if (userInput) {
  userInput.addEventListener('input', autoResizeTextarea);
  userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ---- Boot -----------------------------------------------------------------
renderMessages();

// quick health ping (logs only)
fetch(`${API_BASE}/health`)
  .then(r => r.json())
  .then(d => console.log("[chat.js] Backend health:", d))
  .catch(() => console.warn("[chat.js] Backend not reachable"));

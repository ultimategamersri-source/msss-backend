function sendMessage(userMessage) {
    fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Reply:", data.answer);
    })
    .catch(error => console.error("Error:", error));
}

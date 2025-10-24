from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from vector import load_vector_store
import os
import json
import random
from string import punctuation
from datetime import datetime
import math
import numpy as np  # for matrix, arrays, scientific math
import re
# ----------------------
# Conversational Memory Helpers
# ----------------------
from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="llama3.2")

conversation_history = []
session_memory = []

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stirring-duckanoo-5bd4d6.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def solve_math_expression(expr: str):
    """
    Advanced math solver with:
    - Degree-based trigonometric evaluation (sin(30) = 0.5)
    - Algebraic solving (x+4=10)
    - Scientific computation via SymPy + math
    - Safe eval
    """

    try:
        import sympy as sp
        expr = expr.lower().replace('^', '**').replace('×', '*').replace('÷', '/').strip()

        # Setup symbols and safe environment
        x, y, z = sp.symbols('x y z')
        allowed = {
            "sin": lambda deg: math.sin(math.radians(float(deg))),
            "cos": lambda deg: math.cos(math.radians(float(deg))),
            "tan": lambda deg: math.tan(math.radians(float(deg))),
            "asin": lambda val: math.degrees(math.asin(float(val))),
            "acos": lambda val: math.degrees(math.acos(float(val))),
            "atan": lambda val: math.degrees(math.atan(float(val))),
            "sqrt": math.sqrt,
            "log": math.log10,
            "ln": math.log,
            "pi": math.pi,
            "e": math.e,
            "pow": pow,
        }

        # ✅ Equation solving mode
        if '=' in expr:
            lhs, rhs = expr.split('=')
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            if not solution:
                return "No real solution found."
            if len(solution) == 1:
                return f"The value of x is {solution[0]}."
            return f"Possible values of x are: {', '.join(map(str, solution))}."

        # ✅ Expression simplification
        try:
            simplified = sp.simplify(expr)
            if simplified != expr:
                expr = str(simplified)
        except Exception:
            pass

        # ✅ Safe eval
        result = eval(expr, {"__builtins__": None}, allowed)

        # ✅ Format result cleanly
        if isinstance(result, float):
            result = round(result, 6)
        return f"The result is {result}"

    except Exception as e:
        print(f"⚠️ Math solver error: {e}")
        return None


def explain_math_step_by_step(expr: str):
    """
    Uses sympy to derive, simplify, or solve expressions with step-by-step reasoning.
    Works for equations, derivatives, integrals, simplifications, etc.
    """
    import sympy as sp
    x, y, z = sp.symbols('x y z')

    try:
        # Handle common math keywords
        expr = expr.lower().replace('^', '**').replace('×', '*')
        if "differentiate" in expr or "derivative" in expr or "find dy/dx" in expr:
            target = expr.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.diff(func, x)
            return f"The derivative of {func} with respect to x is: {result}"
        
        elif "integrate" in expr or "integration" in expr:
            target = expr.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.integrate(func, x)
            return f"The integral of {func} with respect to x is: {result} + C"

        elif "=" in expr:
            lhs, rhs = expr.split("=")
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            steps = [
                f"Step 1️⃣: Start with {lhs} = {rhs}",
                f"Step 2️⃣: Move all terms to one side: ({lhs}) - ({rhs}) = 0",
                f"Step 3️⃣: Simplify and solve for x",
                f"✅ Solution: x = {solution}"
            ]
            return "\n".join(steps)

        else:
            simplified = sp.simplify(expr)
            return f"Simplified form: {simplified}"

    except Exception as e:
        return None

def add_to_memory(question: str, answer: str):
    """Add question-answer pair to session memory with embeddings for internal use only."""
    try:
        embed = embedding_model.embed_query(question)
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        embed = None
    session_memory.append({
        "question": question,
        "answer": answer,
        "embedding": embed  # internal only
    })


def retrieve_relevant_memory(question: str, top_n=5):
    """Retrieve top-N relevant past Q&A from session memory using embeddings similarity."""
    try:
        query_embed = embedding_model.embed_query(question)
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return ""

    if not session_memory:
        return ""

    # Compute cosine similarity with each memory embedding
    def cosine_similarity(a, b):
        if a is None or b is None:
            return 0
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(y*y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    scored = []
    for entry in session_memory:
        score = cosine_similarity(query_embed, entry["embedding"])
        scored.append((score, entry))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    top_entries = [f"Q: {e['question']}\nA: {e['answer']}" for _, e in scored[:top_n]]
    return "\n".join(top_entries)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Static assets
# ----------------------
app.mount("/img", StaticFiles(directory="img"), name="img")
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/dist", StaticFiles(directory="dist"), name="dist")

@app.get("/")
def index():
    return FileResponse("index.html")

# ----------------------
# LLMs
# ----------------------
answer_llm = OllamaLLM(model="llama3.2")
emotion_llm = OllamaLLM(model="llama3.2")

# ----------------------
# Manual cache
# ----------------------
MANUAL_CACHE_FILE = "answer_cache.json"

def load_manual_cache():
    try:
        with open(MANUAL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        os.chmod(MANUAL_CACHE_FILE, 0o444)
        return cache
    except Exception as e:
        print(f"⚠️ Manual cache load error: {e}")
        return {}

manual_answer_cache = load_manual_cache()

# ----------------------
# Session Management
# ----------------------
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = None  # Will be created on startup

def start_new_session():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(SESSION_DIR, f"session_{timestamp}.json")
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"🆕 New session started: {session_file}")
    return session_file

def save_session_data(session_file):
    try:
        # Only store question & answer for session file
        data_to_save = [{"question": q["question"], "answer": q["answer"]} for q in session_memory[-50:]]
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save session: {e}")


def cleanup_old_sessions(max_files=10):
    files = sorted(
        [os.path.join(SESSION_DIR, f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
        key=os.path.getmtime
    )
    for f in files[:-max_files]:
        try:
            os.remove(f)
            print(f"🗑️ Deleted old session: {f}")
        except Exception:
            pass

# ----------------------
# Greetings / Farewells
# ----------------------
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
FAREWELLS = ["bye", "goodbye", "see you", "farewell"]

def check_greeting(q: str):
    q = q.lower()
    if any(g in q for g in GREETINGS):
        return "Welcome to Modern Senior Secondary School! I'm Brightly, your assistant. How can I help you today?"
    return None

def check_farewell(q: str):
    q = q.lower()
    if any(f in q for f in FAREWELLS):
        return "Goodbye! Have a great day 🌟 Come back soon!"
    return None

# ----------------------
# Emotion Detection
# ----------------------
def detect_emotion(user_input: str):
    factual_keywords = [
        "what", "where", "when", "how", "who", "which", "fee", "fees",
        "address", "location", "principal", "teacher", "school", "exam",
        "contact", "number", "subject", "student", "class", "admission"
    ]
    if any(re.search(rf"\b{kw}\b", user_input.lower()) for kw in factual_keywords):
        return None
    if any(emoji in user_input for emoji in ["💡", "😊", "😄", "🎉", "🥳"]):
        return None
    prompt = f"""
Detect if this message is Positive (appreciation/humor) or Negative (complaint/anger).
Return only: Positive / Negative / Neutral
Message: {user_input}
"""
    try:
        resp = emotion_llm.invoke(prompt).strip().capitalize()
        if resp == "Positive":
            responses = [
                "That's really kind of you, thank you 😊",
                "Glad to hear that! You're awesome!",
                "That made my day 😄",
                "You're too sweet — thanks a lot!",
                "Aww, I appreciate that 💫"
            ]
            return random.choice(responses)
        elif resp == "Negative":
            return "I'm sorry if something felt off. Let’s fix it together."
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
    return None

# ----------------------
# INTENT MAP
# ----------------------
INTENT_MAP = {
    "fees": ["fee", "fees", "structure", "tuition"],
    "staff": ["principal", "teacher", "staff"],
    "address": ["address", "location", "contact"],
    "self_identity": ["who are you", "your name", "what are you", "who created you"],
}

# ----------------------
# Vector Stores
# ----------------------
vector_stores = {}

def refresh_vector_stores():
    global vector_stores
    current_files = {os.path.splitext(f)[0]: f for f in os.listdir("data") if f.endswith(".txt")}
    for name, file in current_files.items():
        retriever = load_vector_store(file)
        if retriever:
            vector_stores[name] = retriever
    print(f"✅ Vector stores loaded: {list(vector_stores.keys())}")

# ----------------------
# Sub-question Split
# ----------------------
def split_subquestions(q: str):
    if not any(sep in q.lower() for sep in [" and ", ";", "?"]):
        return [q.strip()]
    return [s.strip() for s in re.split(r"[?;]| and ", q) if s.strip()]

# ----------------------
# Memory & Factual Query Helpers
# ----------------------
CLASS_MAP = {
    "lkg": "LKG", "ukg": "UKG", "1st": "I", "first": "I", "i": "I",
    "2nd": "II", "second": "II", "3rd": "III", "third": "III",
    "4th": "IV", "5th": "V", "6th": "VI", "7th": "VII", "8th": "VIII",
    "9th": "IX", "10th": "X", "tenth": "X",
    "11th cs": "XI-CS", "11th bio": "XI-BIO", "11th comm": "XI-COMM",
    "12th cs": "XII-CS", "12th bio": "XII-BIO", "12th comm": "XII-COMM"
}

FEE_KEYWORDS = [
    "tuition", "amenity", "annual", "book",
    "term", "grand total", "total", "fee", "structure"
]

def get_memory_context(question: str, session_memory: list, top_n=3):
    question_lower = question.lower()
    context_pairs = []
    for entry in reversed(session_memory):
        q_lower = entry["question"].lower()
        if any(word in question_lower for word in q_lower.split()):
            context_pairs.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        if len(context_pairs) >= top_n:
            break
    return "\n".join(context_pairs)

def is_factual_query(q: str):
    factual_keywords = [
        "what", "where", "when", "how", "who", "which",
        "fee", "fees", "tuition", "annual", "book", "amenity",
        "term", "total", "structure", "school", "teacher", "principal"
    ]
    return any(kw in q.lower() for kw in factual_keywords)

def get_fee_context(question: str, vector_stores):
    fees_retriever = vector_stores.get("fees")
    if not fees_retriever:
        return ""
    cls = next((v for k, v in CLASS_MAP.items() if k in question.lower()), None)
    docs = fees_retriever.invoke(question) if fees_retriever else []
    if cls:
        docs = [d for d in docs if cls in d.page_content]
    return "\n".join([d.page_content for d in docs])
def safe_retrieve(retriever, query):
    """
    A safe wrapper that works with all LangChain retriever versions.
    Some versions use get_relevant_documents(), others _get_relevant_documents(run_manager=None).
    """
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever._get_relevant_documents(query, run_manager=None)
# ----------------------
NCERT_DATASETS = {
    "ncert_maths": "data/ncert_maths.txt",
    "ncert_physics": "data/ncert_physics.txt",
    "ncert_chemistry": "data/ncert_chemistry.txt",
}

def load_ncert_vectors():
    for name, path in NCERT_DATASETS.items():
        if os.path.exists(path):
            retriever = load_vector_store(path)
            if retriever:
                vector_stores[name] = retriever
    print(f"📘 NCERT datasets loaded: {list(NCERT_DATASETS.keys())}")
math_regex = re.compile(
    r"\b(d/dx|dy/dx|dx|differentiate|derive|integrate|∫|roots?|equation|simplify|factor|solve|"
    r"sin|cos|tan|log|ln|sqrt|exp|pi|e)\b|[\d+\-*/^()=]", re.IGNORECASE
)


# ----------------------
# Query Model
# ----------------------
class Query(BaseModel):
    question: str

# ----------------------
# Ask endpoint
# ----------------------
@app.post("/ask")
async def ask(query: Query):
    global SESSION_FILE
    q_text = query.question.strip()
    final_answers = []

    # Math regex
    math_regex = re.compile(r"d/dx|dx|differentiate|derive|integrate|roots|equation|simplify|sin|cos|tan|log|sqrt|=|[\d+\-*/^()]")

    # 1️⃣ Math & Science shortcut on full question
    if math_regex.search(q_text):
        step_result = explain_math_step_by_step(q_text)
        if step_result:
            add_to_memory(q_text, step_result)
            conversation_history.append({"question": q_text, "answer": step_result})
            return JSONResponse({"answer": step_result, "history": conversation_history})

        math_result = solve_math_expression(q_text)
        if math_result:
            add_to_memory(q_text, math_result)
            conversation_history.append({"question": q_text, "answer": math_result})
            return JSONResponse({"answer": math_result, "history": conversation_history})

    # 2️⃣ Greeting / Farewell / Emotion
    if resp := check_greeting(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := check_farewell(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := detect_emotion(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})

    # 3️⃣ Self-identity / capabilities
    answer = None
    lower_q = q_text.lower()
    if any(phrase in lower_q for phrase in INTENT_MAP["self_identity"]):
        answer = "I'm Brightly — your friendly Modern Senior Secondary School assistant..."
    elif any(word in lower_q for word in ["provide", "offer", "help", "assist", "what can you"]):
        answer = random.choice([
            "I can help you with school details, fees, admissions, exams, and staff information.",
            "I assist with queries about Modern Senior Secondary School — like fees, staff, or classes.",
            "I provide details about school activities, admissions, and academic info.",
            "I’m here to share school-related information and help you find what you need!"
        ])

    if answer:
        add_to_memory(q_text, answer)
        conversation_history.append({"question": q_text, "answer": answer})
        if not SESSION_FILE:
            SESSION_FILE = start_new_session()
        save_session_data(SESSION_FILE)
        cleanup_old_sessions(max_files=10)
        history_for_frontend = [{"question": q["question"], "answer": q["answer"]} for q in conversation_history]
        return JSONResponse({"answer": answer, "history": history_for_frontend})

    # 4️⃣ Main query processing (split sub-questions)
    sub_qs = split_subquestions(q_text)

    simple_math_questions = {
        "quadratic equations": "A quadratic equation is of the form ax² + bx + c = 0, ... examples ..."
    }

    for sq in sub_qs:
        answer = None

        # Check simple math FAQs
        for key, val in simple_math_questions.items():
            if key in sq.lower():
                answer = val
                break

        # Math / science shortcut
        if not answer and math_regex.search(sq):
            step_result = explain_math_step_by_step(sq)
            answer = step_result or solve_math_expression(sq)

        # LLM / vector / memory fallback
        if not answer:
            context = ""
            if manual_resp := manual_answer_cache.get(sq):
                context += manual_resp["answer"] + "\n"
            for store_name, retriever in vector_stores.items():
                results = safe_retrieve(retriever, sq)
                if results:
                    context += "\n".join([doc.page_content for doc in results]) + "\n"
            conv_context = retrieve_relevant_memory(sq)
            if conv_context:
                context += "\n--- Previous conversation ---\n" + conv_context
            if not context.strip():
                context = "No data found."

            prompt = f"""
You are called and are Brightly — the official AI assistant of Modern Senior Secondary School, Chennai.
You are also the greatest school AI tutor.
You can explain and solve problems in math, physics, and chemistry like an experienced teacher.
You can perform calculations, algebra, and trigonometry using your internal math tools.
When students ask a study question:
- Give short but clear explanations with steps.
- Avoid unnecessary emojis or robotic tone.
- Use plain, teacher-style English.

Knowledge Scope:
- NCERT-based Physics, Chemistry, Maths (Classes 6–12)
- General school information, guidance, and basic academics
- You can reason, calculate, and explain formulas clearly

Tone:
- Friendly, conversational, natural
- Humble and context-aware
- Avoid robotic style

Rules:
-avoid political talks and if something is unrelated to school studies and school avoid it
- Never say you were created by OpenAI or Meta
- If asked who created you, answer: "I was created by the technical team at Modern Senior Secondary School to assist students and parents with school-related queries."
- If irrelevant/out-of-scope, respond naturally.
- Keep responses short and human-like.
- You are an Indian English speaker. Maintain cultural relevance and ruppees.

Context:
{context}
Question: {sq}
Answer:
"""
            try:
                answer = answer_llm.invoke(prompt).strip()
            except Exception as e:
                answer = "I’m having trouble accessing the data at the moment, please try again."
                print(f"⚠️ LLM error: {e}")

        # Store memory + conversation
        add_to_memory(sq, answer)
        conversation_history.append({"question": sq, "answer": answer})
        final_answers.append(answer)

    # Save session after all sub-questions
    if not SESSION_FILE:
        SESSION_FILE = start_new_session()
    save_session_data(SESSION_FILE)
    cleanup_old_sessions(max_files=10)

    history_for_frontend = [{"question": q["question"], "answer": q["answer"]} for q in conversation_history]
    return JSONResponse({"answer": "\n".join(final_answers), "history": history_for_frontend})

# ----------------------
# Startup & Shutdown
# ----------------------
@app.on_event("startup")
def startup_event():
    print("🚀 Server starting up...")
    cleanup_old_sessions(max_files=10)
    refresh_vector_stores()
    load_ncert_vectors()
    print("✅ Vector stores + NCERT data loaded.")

@app.on_event("shutdown")
def shutdown_event():
    print("🚪 Server shutting down...")
    if SESSION_FILE:
        save_session_data(SESSION_FILE)
        cleanup_old_sessions(max_files=10)
        print(f"💾 Session data saved to {SESSION_FILE}")

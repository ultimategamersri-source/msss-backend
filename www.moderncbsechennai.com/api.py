import os
import json
import math
import re
import random
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ----------------------
# Defaults for your deployment (can be overridden by env)
# ----------------------
DEFAULT_PROJECT = "vertex-ai-search-rag-project"
DEFAULT_LOCATION = "asia-south1"

def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

GOOGLE_CLOUD_PROJECT = env("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
GOOGLE_CLOUD_LOCATION = env("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
REFRESH_VECTORS_ON_STARTUP = env("REFRESH_VECTORS_ON_STARTUP", "true").lower() == "true"

# ----------------------
# Vector store loader (ours). Keep import inside try so app still boots if file missing.
# ----------------------
try:
    from vector import load_vector_store
except Exception as e:
    load_vector_store = None
    print(f"ℹ️ vector.py not available or failed to import: {e}")

# LangChain + Vertex AI (lazy constructors below prevent init at import time)
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

# ----------------------
# Helpers: lazy constructors (avoid init-at-import crashes on Cloud Run)
# ----------------------
def get_embedding_model():
    # Construct only on first use, pulling project/region from env/defaults
    return VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

_answer_llm = None
_emotion_llm = None

def get_answer_llm():
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = VertexAI(
            model_name="gemini-1.5-flash",
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    return _answer_llm

def get_emotion_llm():
    global _emotion_llm
    if _emotion_llm is None:
        _emotion_llm = VertexAI(
            model_name="gemini-1.5-flash",
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    return _emotion_llm

# ----------------------
# Globals / state
# ----------------------
conversation_history = []
session_memory = []

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
for _d in ("img", "css", "dist"):
    os.makedirs(_d, exist_ok=True)

# ----------------------
# Single FastAPI instance
# ----------------------
app = FastAPI(title="MSSS Backend", version="1.0.0")

# CORS setup (widen as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static if present
if os.path.isdir("img"):
    app.mount("/img", StaticFiles(directory="img"), name="img")
if os.path.isdir("css"):
    app.mount("/css", StaticFiles(directory="css"), name="css")
if os.path.isdir("dist"):
    app.mount("/dist", StaticFiles(directory="dist"), name="dist")

@app.get("/")
def index():
    # Return index.html if present, else a simple JSON (prevents boot failure)
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({
        "message": "Modern Senior Secondary School — API online",
        "project": GOOGLE_CLOUD_PROJECT,
        "location": GOOGLE_CLOUD_LOCATION,
        "vectors_refreshed_on_startup": REFRESH_VECTORS_ON_STARTUP
    })

@app.get("/health")
@app.get("/_/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(payload: dict):
    msg = payload.get("message", "")
    return {"reply": f"Echo: {msg}"}

# ----------------------
# Math utilities
# ----------------------
def solve_math_expression(expr: str):
    """
    Degree-based trig; simple equation solving with SymPy; safe eval sandbox.
    """
    try:
        import sympy as sp

        expr = (
            expr.lower()
            .replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .strip()
        )

        x, y, z = sp.symbols("x y z")
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

        if "=" in expr:
            lhs, rhs = expr.split("=")
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            if not solution:
                return "No real solution found."
            if len(solution) == 1:
                return f"The value of x is {solution[0]}."
            return f"Possible values of x are: {', '.join(map(str, solution))}."

        try:
            simplified = sp.simplify(expr)
            if str(simplified) != expr:
                expr = str(simplified)
        except Exception:
            pass

        result = eval(expr, {"__builtins__": None}, allowed)  # noqa: S307 (guarded)
        if isinstance(result, float):
            result = round(result, 6)
        return f"The result is {result}"

    except Exception as e:
        print(f"⚠️ Math solver error: {e}")
        return None


def explain_math_step_by_step(expr: str):
    import sympy as sp
    x, y, z = sp.symbols("x y z")
    try:
        expr = expr.lower().replace("^", "**").replace("×", "*")
        if ("differentiate" in expr) or ("derivative" in expr) or ("find dy/dx" in expr):
            target = expr.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.diff(func, x)
            return f"The derivative of {func} with respect to x is: {result}"
        elif ("integrate" in expr) or ("integration" in expr):
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
                f"✅ Solution: x = {solution}",
            ]
            return "\n".join(steps)
        else:
            simplified = sp.simplify(expr)
            return f"Simplified form: {simplified}"
    except Exception:
        return None

# ----------------------
# Memory helpers
# ----------------------
def add_to_memory(question: str, answer: str):
    try:
        embed = get_embedding_model().embed_query(question)
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        embed = None
    session_memory.append({"question": question, "answer": answer, "embedding": embed})

def retrieve_relevant_memory(question: str, top_n=5):
    try:
        query_embed = get_embedding_model().embed_query(question)
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return ""

    if not session_memory:
        return ""

    def cosine_similarity(a, b):
        if a is None or b is None:
            return 0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
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

# ----------------------
# Greetings / Farewells / Emotion
# ----------------------
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
FAREWELLS = ["bye", "goodbye", "see you", "farewell"]

def check_greeting(q: str):
    q = q.lower()
    if any(g in q for g in GREETINGS):
        return ("Welcome to Modern Senior Secondary School! I'm Brightly, your assistant. "
                "How can I help you today?")
    return None

def check_farewell(q: str):
    q = q.lower()
    if any(f in q for f in FAREWELLS):
        return "Goodbye! Have a great day 🌟 Come back soon!"
    return None

def detect_emotion(user_input: str):
    factual_keywords = [
        "what","where","when","how","who","which","fee","fees","address","location",
        "principal","teacher","school","exam","contact","number","subject","student",
        "class","admission",
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
        resp = get_emotion_llm().invoke(prompt).strip().capitalize()
        if resp == "Positive":
            responses = [
                "That's really kind of you, thank you 😊",
                "Glad to hear that! You're awesome!",
                "That made my day 😄",
                "You're too sweet — thanks a lot!",
                "Aww, I appreciate that 💫",
            ]
            return random.choice(responses)
        elif resp == "Negative":
            return "I'm sorry if something felt off. Let’s fix it together."
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
    return None

# ----------------------
# Intent & Vector Stores
# ----------------------
INTENT_MAP = {
    "fees": ["fee", "fees", "structure", "tuition"],
    "staff": ["principal", "teacher", "staff"],
    "address": ["address", "location", "contact"],
    "self_identity": ["who are you", "your name", "what are you", "who created you"],
}

vector_stores = {}

def refresh_vector_stores():
    """Build retrievers from all .txt files in ./data (if present)."""
    global vector_stores
    vector_stores = {}
    if load_vector_store is None:
        print("ℹ️ load_vector_store unavailable; skipping vector build.")
        return
    if not os.path.isdir("data"):
        print("ℹ️ No data directory found; skipping vector build.")
        return
    current_files = {
        os.path.splitext(f)[0]: os.path.join("data", f)
        for f in os.listdir("data")
        if f.endswith(".txt")
    }
    for name, file in current_files.items():
        try:
            retriever = load_vector_store(file)
            if retriever:
                vector_stores[name] = retriever
        except Exception as e:
            print(f"⚠️ Failed to build retriever for '{file}': {e}")
    print(f"✅ Vector stores loaded: {list(vector_stores.keys())}")

# ----------------------
# Misc helpers
# ----------------------
def split_subquestions(q: str):
    if not any(sep in q.lower() for sep in [" and ", ";", "?"]):
        return [q.strip()]
    return [s.strip() for s in re.split(r"[?;]| and ", q) if s.strip()]

CLASS_MAP = {
    "lkg": "LKG", "ukg": "UKG", "1st": "I", "first": "I", "i": "I",
    "2nd": "II", "second": "II", "3rd": "III", "third": "III",
    "4th": "IV", "5th": "V", "6th": "VI", "7th": "VII", "8th": "VIII",
    "9th": "IX", "10th": "X", "tenth": "X",
    "11th cs": "XI-CS", "11th bio": "XI-BIO", "11th comm": "XI-COMM",
    "12th cs": "XII-CS", "12th bio": "XII-BIO", "12th comm": "XII-COMM",
}

def safe_retrieve(retriever, query):
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever._get_relevant_documents(query, run_manager=None)

# ----------------------
# NCERT optional datasets (if present)
# ----------------------
NCERT_DATASETS = {
    "ncert_maths": "data/ncert_maths.txt",
    "ncert_physics": "data/ncert_physics.txt",
    "ncert_chemistry": "data/ncert_chemistry.txt",
}

def load_ncert_vectors():
    if load_vector_store is None:
        print("ℹ️ load_vector_store unavailable; skipping NCERT.")
        return
    loaded = []
    for name, path in NCERT_DATASETS.items():
        if os.path.exists(path):
            try:
                retriever = load_vector_store(path)
                if retriever:
                    vector_stores[name] = retriever
                    loaded.append(name)
            except Exception as e:
                print(f"⚠️ Failed to load NCERT '{name}': {e}")
    print(f"📘 NCERT datasets loaded: {loaded}")

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
    q_text = query.question.strip()
    final_answers = []

    math_regex = re.compile(
        r"d/dx|dx|differentiate|derive|integrate|roots|equation|simplify|sin|cos|tan|log|sqrt|=|[\d+\-*/^()]"
    )

    # 1) Math / Science direct
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

    # 2) Greetings / Farewell / Emotion
    if resp := check_greeting(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := check_farewell(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := detect_emotion(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})

    # 3) Identity / capabilities
    answer = None
    lower_q = q_text.lower()
    if any(phrase in lower_q for phrase in INTENT_MAP["self_identity"]):
        answer = "I'm Brightly — your friendly Modern Senior Secondary School assistant..."
    elif any(word in lower_q for word in ["provide", "offer", "help", "assist", "what can you"]):
        answer = random.choice(
            [
                "I can help you with school details, fees, admissions, exams, and staff information.",
                "I assist with queries about Modern Senior Secondary School — like fees, staff, or classes.",
                "I provide details about school activities, admissions, and academic info.",
                "I’m here to share school-related information and help you find what you need!",
            ]
        )

    if answer:
        add_to_memory(q_text, answer)
        conversation_history.append({"question": q_text, "answer": answer})
        return JSONResponse({"answer": answer, "history": conversation_history})

    # 4) Main processing (split)
    sub_qs = split_subquestions(q_text)

    simple_math_questions = {
        "quadratic equations": "A quadratic equation is of the form ax² + bx + c = 0, ... examples ..."
    }

    for sq in sub_qs:
        answer = None

        for key, val in simple_math_questions.items():
            if key in sq.lower():
                answer = val
                break

        if not answer and math_regex.search(sq):
            step_result = explain_math_step_by_step(sq)
            answer = step_result or solve_math_expression(sq)

        if not answer:
            context = ""
            # Manual cache (optional file)
            MANUAL_CACHE_FILE = "answer_cache.json"
            if os.path.exists(MANUAL_CACHE_FILE):
                try:
                    with open(MANUAL_CACHE_FILE, "r", encoding="utf-8") as f:
                        manual_cache = json.load(f)
                    if sq in manual_cache:
                        context += str(manual_cache[sq].get("answer", "")) + "\n"
                except Exception as e:
                    print(f"⚠️ Manual cache load error: {e}")

            # Vectors
            for store_name, retriever in vector_stores.items():
                try:
                    results = safe_retrieve(retriever, sq)
                    if results:
                        context += "\n".join([doc.page_content for doc in results]) + "\n"
                except Exception as e:
                    print(f"⚠️ Retriever '{store_name}' error: {e}")

            # Conversation memory
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
- avoid political talks and if something is unrelated to school studies and school avoid it
- Never say you were created by OpenAI or Meta
- If asked who created you, answer: "I was created by the technical team at Modern Senior Secondary School to assist students and parents with school-related queries."
- If irrelevant/out-of-scope, respond naturally.
- Keep responses short and human-like.
- You are an Indian English speaker. Maintain cultural relevance and rupees.

Context:
{context}
Question: {sq}
Answer:
"""
            try:
                answer = get_answer_llm().invoke(prompt).strip()
            except Exception as e:
                print(f"⚠️ LLM error: {e}")
                answer = "I’m having trouble accessing the data at the moment, please try again."

        add_to_memory(sq, answer)
        conversation_history.append({"question": sq, "answer": answer})
        final_answers.append(answer)

    return JSONResponse({"answer": "\n".join(final_answers), "history": conversation_history})

# ----------------------
# Session housekeeping (optional persistence)
# ----------------------
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = None

def start_new_session():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SESSION_DIR, f"session_{timestamp}.json")

def save_session_data(session_file):
    try:
        data_to_save = [{"question": q["question"], "answer": q["answer"]} for q in session_memory[-50:]]
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save session: {e}")

def cleanup_old_sessions(max_files=10):
    try:
        files = sorted(
            [os.path.join(SESSION_DIR, f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
            key=os.path.getmtime,
        )
        for f in files[:-max_files]:
            try:
                os.remove(f)
                print(f"🗑️ Deleted old session: {f}")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Session cleanup error: {e}")

# ----------------------
# Startup / Shutdown
# ----------------------
@app.on_event("startup")
def startup_event():
    print("🚀 Server starting up...")
    print(f"   Project = {GOOGLE_CLOUD_PROJECT}")
    print(f"   Location = {GOOGLE_CLOUD_LOCATION}")
    print(f"   Refresh vectors on startup = {REFRESH_VECTORS_ON_STARTUP}")
    cleanup_old_sessions(max_files=10)

    if REFRESH_VECTORS_ON_STARTUP:
        refresh_vector_stores()
        # Optional NCERT files if present
        load_ncert_vectors()
        print("✅ Vector stores + NCERT data loaded.")
    else:
        print("⏭️ Skipping vector refresh/load on startup.")

@app.on_event("shutdown")
def shutdown_event():
    print("🚪 Server shutting down...")
    global SESSION_FILE
    if SESSION_FILE is None:
        SESSION_FILE = start_new_session()
    save_session_data(SESSION_FILE)
    cleanup_old_sessions(max_files=10)
    print(f"💾 Session data saved to {SESSION_FILE}")

# Local dev entrypoint (Cloud Run ignores this, uses your Docker CMD)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)

# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, json, math, re, random
from datetime import datetime
from typing import List, Dict, Any

# ----------------------
# App & CORS
# ----------------------
app = FastAPI(title="MSSS Backend", version="1.0.0")

NETLIFY_ORIGIN = os.getenv("NETLIFY_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_ORIGIN] if NETLIFY_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Health / Smoke
# ----------------------
@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(payload: dict):
    msg = payload.get("message", "")
    return {"reply": f"Echo: {msg}"}

# ----------------------
# Globals / Memory
# ----------------------
conversation_history: List[Dict[str, Any]] = []
session_memory: List[Dict[str, Any]] = []

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
os.makedirs("data", exist_ok=True)  # in case data is added later

SESSION_DIR = "sessions"
MANUAL_CACHE_FILE = "answer_cache.json"
SESSION_FILE = None

# ----------------------
# LangChain/Ollama (lazy)
# ----------------------
_answer_llm = None
_emotion_llm = None
_embedding_model = None
_ollama_ready = False

def ensure_ollama():
    """Attempt once to connect to Ollama. Never block startup."""
    global _answer_llm, _emotion_llm, _embedding_model, _ollama_ready
    if _ollama_ready:
        return True
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings  # type: ignore
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        _answer_llm = OllamaLLM(model=model)
        _emotion_llm = OllamaLLM(model=model)
        _embedding_model = OllamaEmbeddings(model=model)
        _ollama_ready = True
        print("[ollama] connected")
    except Exception as e:
        print(f"[ollama] not available: {e}")
        _ollama_ready = False
    return _ollama_ready

# ----------------------
# Math helpers
# ----------------------
def solve_math_expression(expr: str):
    """
    Degree-based trig, algebraic solving, simplification.
    """
    try:
        import sympy as sp
        expr = expr.lower().replace("^", "**").replace("×", "*").replace("÷", "/").strip()
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
        if '=' in expr:
            lhs, rhs = expr.split('=')
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
        result = eval(expr, {"__builtins__": None}, allowed)
        if isinstance(result, float):
            result = round(result, 6)
        return f"The result is {result}"
    except Exception as e:
        print(f"[math] error: {e}")
        return None

def explain_math_step_by_step(expr: str):
    """
    Uses sympy to derive, integrate, solve, or simplify and produce a brief explanation.
    """
    try:
        import sympy as sp
        x, y, z = sp.symbols('x y z')
        t = expr.lower().replace('^', '**').replace('×', '*')

        if any(k in t for k in ["differentiate", "derivative", "find dy/dx"]):
            target = t.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.diff(func, x)
            return f"The derivative of {func} with respect to x is: {result}"

        if "integrate" in t or "integration" in t:
            target = t.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.integrate(func, x)
            return f"The integral of {func} with respect to x is: {result} + C"

        if "=" in t:
            lhs, rhs = t.split("=")
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            steps = [
                f"Start with {lhs} = {rhs}",
                f"Move all terms to one side: ({lhs}) - ({rhs}) = 0",
                f"Solve for x",
                f"Solution: x = {solution}"
            ]
            return "\n".join(steps)

        simplified = sp.simplify(t)
        return f"Simplified form: {simplified}"
    except Exception:
        return None

# ----------------------
# Memory helpers
# ----------------------
def add_to_memory(question: str, answer: str):
    embed = None
    if ensure_ollama() and _embedding_model:
        try:
            embed = _embedding_model.embed_query(question)
        except Exception as e:
            print(f"[embed] error: {e}")
    session_memory.append({"question": question, "answer": answer, "embedding": embed})

def retrieve_relevant_memory(question: str, top_n=5):
    if not session_memory or not ensure_ollama() or not _embedding_model:
        return ""
    try:
        query_embed = _embedding_model.embed_query(question)
    except Exception as e:
        print(f"[embed] query error: {e}")
        return ""

    def cosine(a, b):
        if a is None or b is None:
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = sum(x*x for x in a) ** 0.5
        nb = sum(y*y for y in b) ** 0.5
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    scored = [(cosine(query_embed, e["embedding"]), e) for e in session_memory]
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [f"Q: {e['question']}\nA: {e['answer']}" for _, e in scored[:top_n]]
    return "\n".join(top)

# ----------------------
# Greetings / Farewells / Emotion
# ----------------------
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
FAREWELLS = ["bye", "goodbye", "see you", "farewell"]

def check_greeting(q: str):
    ql = q.lower()
    if any(g in ql for g in GREETINGS):
        return "Welcome to Modern Senior Secondary School! I'm Brightly, your assistant. How can I help you today?"
    return None

def check_farewell(q: str):
    ql = q.lower()
    if any(f in ql for f in FAREWELLS):
        return "Goodbye! Have a great day 🌟 Come back soon!"
    return None

def detect_emotion(user_input: str):
    # Avoid LLM call for obvious factual queries
    factual_keywords = [
        "what","where","when","how","who","which","fee","fees","address",
        "location","principal","teacher","school","exam","contact","number",
        "subject","student","class","admission"
    ]
    if any(re.search(rf"\b{kw}\b", user_input.lower()) for kw in factual_keywords):
        return None
    if any(emoji in user_input for emoji in ["💡", "😊", "😄", "🎉", "🥳"]):
        return None

    if ensure_ollama() and _emotion_llm:
        prompt = f"Detect if this message is Positive / Negative / Neutral:\nMessage: {user_input}"
        try:
            resp = _emotion_llm.invoke(prompt).strip().capitalize()
            if resp == "Positive":
                return random.choice([
                    "That's really kind of you, thank you 😊",
                    "Glad to hear that! You're awesome!",
                    "That made my day 😄",
                    "You're too sweet — thanks a lot!",
                    "Aww, I appreciate that 💫"
                ])
            if resp == "Negative":
                return "I'm sorry if something felt off. Let’s fix it together."
        except Exception as e:
            print(f"[emotion] error: {e}")
    return None

# ----------------------
# Intent map
# ----------------------
INTENT_MAP = {
    "fees": ["fee", "fees", "structure", "tuition"],
    "staff": ["principal", "teacher", "staff"],
    "address": ["address", "location", "contact"],
    "self_identity": ["who are you", "your name", "what are you", "who created you"],
}

# ----------------------
# Vector store loading
# ----------------------
def safe_import_vector_loader():
    try:
        from vector import load_vector_store  # your existing helper
        return load_vector_store
    except Exception as e:
        print(f"[vector] loader unavailable: {e}")
        return None

vector_stores: Dict[str, Any] = {}

def refresh_vector_stores():
    load_fn = safe_import_vector_loader()
    if not load_fn:
        return
    if not os.path.isdir("data"):
        return
    try:
        current_files = {os.path.splitext(f)[0]: f for f in os.listdir("data") if f.endswith(".txt")}
        for name, file in current_files.items():
            retriever = load_fn(os.path.join("data", file))
            if retriever:
                vector_stores[name] = retriever
        print(f"[vector] loaded: {list(vector_stores.keys())}")
    except Exception as e:
        print(f"[vector] refresh error: {e}")

def safe_retrieve(retriever, query):
    if not retriever:
        return []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        return retriever._get_relevant_documents(query, run_manager=None)  # type: ignore
    except Exception as e:
        print(f"[vector] retrieve error: {e}")
        return []

NCERT_DATASETS = {
    "ncert_maths": "data/ncert_maths.txt",
    "ncert_physics": "data/ncert_physics.txt",
    "ncert_chemistry": "data/ncert_chemistry.txt",
}

def load_ncert_vectors():
    load_fn = safe_import_vector_loader()
    if not load_fn:
        return
    for name, path in NCERT_DATASETS.items():
        if os.path.exists(path):
            try:
                retriever = load_fn(path)
                if retriever:
                    vector_stores[name] = retriever
            except Exception as e:
                print(f"[vector] NCERT load error {name}: {e}")
    print(f"[vector] NCERT datasets configured: {list(NCERT_DATASETS.keys())}")

# ----------------------
# Split / Classes / Fees helpers
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
    "12th cs": "XII-CS", "12th bio": "XII-BIO", "12th comm": "XII-COMM"
}

FEE_KEYWORDS = [
    "tuition", "amenity", "annual", "book",
    "term", "grand total", "total", "fee", "structure"
]

def get_fee_context(question: str, vector_stores: Dict[str, Any]):
    fees_retriever = vector_stores.get("fees")
    if not fees_retriever:
        return ""
    cls = next((v for k, v in CLASS_MAP.items() if k in question.lower()), None)
    docs = safe_retrieve(fees_retriever, question)
    if cls:
        docs = [d for d in docs if hasattr(d, "page_content") and cls in d.page_content]
    return "\n".join([d.page_content for d in docs if hasattr(d, "page_content")])

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
        "what","where","when","how","who","which",
        "fee","fees","tuition","annual","book","amenity",
        "term","total","structure","school","teacher","principal"
    ]
    return any(kw in q.lower() for kw in factual_keywords)

# ----------------------
# Manual cache
# ----------------------
def load_manual_cache():
    try:
        if not os.path.exists(MANUAL_CACHE_FILE):
            return {}
        with open(MANUAL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        try:
            os.chmod(MANUAL_CACHE_FILE, 0o444)
        except Exception:
            pass
        return cache
    except Exception as e:
        print(f"[cache] load error: {e}")
        return {}

manual_answer_cache = load_manual_cache()

# ----------------------
# Sessions
# ----------------------
def start_new_session():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(SESSION_DIR, f"session_{timestamp}.json")
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"[session] new: {session_file}")
    except Exception as e:
        print(f"[session] create error: {e}")
    return session_file

def save_session_data(session_file):
    if not session_file:
        return
    try:
        data_to_save = [{"question": q["question"], "answer": q["answer"]} for q in session_memory[-50:]]
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[session] save error: {e}")

def cleanup_old_sessions(max_files=10):
    try:
        files = sorted(
            [os.path.join(SESSION_DIR, f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
            key=os.path.getmtime
        )
        for f in files[:-max_files]:
            try:
                os.remove(f)
                print(f"[session] removed: {f}")
            except Exception:
                pass
    except Exception as e:
        print(f"[session] cleanup error: {e}")

# ----------------------
# Ask endpoint (full flow)
# ----------------------
class Query(BaseModel):
    question: str

_math_rx = re.compile(
    r"\b(d/dx|dy/dx|dx|differentiate|derive|integrate|∫|roots?|equation|simplify|factor|solve|"
    r"sin|cos|tan|log|ln|sqrt|exp|pi|e)\b|[\d+\-*/^()=]",
    re.IGNORECASE
)

@app.post("/ask")
async def ask(query: Query):
    global SESSION_FILE
    q_text = (query.question or "").strip()
    if not q_text:
        return JSONResponse({"answer": "Please type a question.", "history": conversation_history})

    # Math/science shortcut on full question
    if _math_rx.search(q_text):
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

    # Greeting / Farewell / Emotion
    if resp := check_greeting(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := check_farewell(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := detect_emotion(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})

    # Self-identity / capabilities
    lower_q = q_text.lower()
    base_answer = None
    if any(phrase in lower_q for phrase in INTENT_MAP["self_identity"]):
        base_answer = (
            "I'm Brightly — your friendly Modern Senior Secondary School assistant. "
            "I was created by the technical team at Modern Senior Secondary School to help with school-related queries."
        )
    elif any(w in lower_q for w in ["provide", "offer", "help", "assist", "what can you"]):
        base_answer = random.choice([
            "I can help you with school details, fees, admissions, exams, and staff information.",
            "I assist with queries about Modern Senior Secondary School — like fees, staff, or classes.",
            "I provide details about school activities, admissions, and academic info.",
            "I’m here to share school-related information and help you find what you need!"
        ])
    if base_answer:
        add_to_memory(q_text, base_answer)
        conversation_history.append({"question": q_text, "answer": base_answer})
        if not SESSION_FILE:
            SESSION_FILE = start_new_session()
        save_session_data(SESSION_FILE)
        cleanup_old_sessions(max_files=10)
        return JSONResponse({"answer": base_answer, "history": conversation_history})

    # Main query processing (split sub-questions)
    sub_qs = split_subquestions(q_text)
    final_answers: List[str] = []

    # simple math FAQs (extensible)
    simple_math_questions = {
        "quadratic equations": "A quadratic equation has the form ax² + bx + c = 0. Its solutions are x = [-b ± √(b²−4ac)] / (2a)."
    }

    for sq in sub_qs:
        answer = None

        # FAQs
        for key, val in simple_math_questions.items():
            if key in sq.lower():
                answer = val
                break

        # Math shortcut per sub-question
        if not answer and _math_rx.search(sq):
            step_result = explain_math_step_by_step(sq)
            answer = step_result or solve_math_expression(sq)

        # LLM / vector / memory fallback
        if not answer:
            context = ""
            if manual_resp := manual_answer_cache.get(sq):
                if isinstance(manual_resp, dict) and "answer" in manual_resp:
                    context += str(manual_resp["answer"]) + "\n"
                elif isinstance(manual_resp, str):
                    context += manual_resp + "\n"

            # vector stores
            for store_name, retriever in vector_stores.items():
                results = safe_retrieve(retriever, sq)
                if results:
                    context += "\n".join([getattr(doc, "page_content", "") for doc in results if hasattr(doc, "page_content")]) + "\n"

            # conversational memory
            conv_context = retrieve_relevant_memory(sq)
            if conv_context:
                context += "\n--- Previous conversation ---\n" + conv_context

            if not context.strip():
                context = "No data found."

            # LLM if available
            if ensure_ollama() and _answer_llm:
                prompt = f"""
You are Brightly — the official AI assistant of Modern Senior Secondary School, Chennai.
You are also a capable school AI tutor for maths/physics/chemistry.
Keep answers short, clear, teacher-style Indian English. Avoid politics and off-topic.

Context:
{context}

Question: {sq}
Answer:
""".strip()
                try:
                    answer = _answer_llm.invoke(prompt).strip()
                except Exception as e:
                    print(f"[llm] error: {e}")
                    answer = None

            # final fallback text
            if not answer:
                answer = "I’m having trouble accessing the AI model or data right now. Please try again or ask a math question like ‘integrate x^2’."

        add_to_memory(sq, answer)
        conversation_history.append({"question": sq, "answer": answer})
        final_answers.append(answer)

    if not SESSION_FILE:
        SESSION_FILE = start_new_session()
    save_session_data(SESSION_FILE)
    cleanup_old_sessions(max_files=10)

    return JSONResponse({"answer": "\n".join(final_answers), "history": conversation_history})

# ----------------------
# Startup / Shutdown
# ----------------------
@app.on_event("startup")
def startup_event():
    print("🚀 Server starting...")
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
        # Try Ollama (non-blocking)
        ensure_ollama()
        # Vector stores
        refresh_vector_stores()
        load_ncert_vectors()
        print("✅ Startup complete.")
    except Exception as e:
        print(f"[startup] error: {e}")

@app.on_event("shutdown")
def shutdown_event():
    print("🚪 Server shutting down...")
    try:
        if SESSION_FILE:
            save_session_data(SESSION_FILE)
            cleanup_old_sessions(max_files=10)
    except Exception as e:
        print(f"[shutdown] error: {e}")

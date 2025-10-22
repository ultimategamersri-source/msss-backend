import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import shutil

VECTOR_DIR = "vectorstore"
DATA_DIR = "data"

def build_vector_store(file_name: str):
    base_name = os.path.splitext(file_name)[0]
    data_path = os.path.join(DATA_DIR, file_name)
    vector_path = os.path.join(VECTOR_DIR, base_name)

    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}, skipping vector creation.")
        return None

    print(f"📘 Loading data from: {data_path}")
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()

    embeddings = OllamaEmbeddings(model="llama3.2")
    print(f"⚙️ Creating FAISS index for '{base_name}'...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(vector_path, exist_ok=True)
    vectorstore.save_local(vector_path)
    print(f"✅ Vector store saved for '{base_name}' at {vector_path}")
    return vectorstore

def load_vector_store(file_name: str):
    if not file_name.endswith(".txt"):
        file_name += ".txt"

    base_name = os.path.splitext(file_name)[0]
    data_path = os.path.join(DATA_DIR, file_name)
    vector_path = os.path.join(VECTOR_DIR, base_name)
    index_file = os.path.join(vector_path, "index.faiss")

    embeddings = OllamaEmbeddings(model="llama3.2")

    if not os.path.exists(data_path):
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
            print(f"🗑️ Deleted vector store for missing file '{base_name}'")
        return None

    if (not os.path.exists(vector_path) or
        not os.path.exists(index_file) or
        os.path.getmtime(data_path) > os.path.getmtime(index_file)):
        print(f"⚠️ Rebuilding vector store for '{base_name}'...")
        build_vector_store(file_name)

    if not os.path.exists(index_file):
        print(f"❌ Vector store missing for '{base_name}', skipping.")
        return None

    print(f"📦 Loading FAISS vector store for '{base_name}'...")
    db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ '{base_name}' vector loaded.")
    return db.as_retriever(search_kwargs={"k": 5})

def load_all_retrievers():
    retrievers = {}
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory missing: {DATA_DIR}")
        return retrievers

    current_files = {os.path.splitext(f)[0]: f for f in os.listdir(DATA_DIR) if f.endswith(".txt")}

    # Remove stale vector stores for deleted files
    if os.path.exists(VECTOR_DIR):
        for folder in os.listdir(VECTOR_DIR):
            if folder not in current_files:
                shutil.rmtree(os.path.join(VECTOR_DIR, folder))
                print(f"🗑️ Removed stale vector store '{folder}'")

    # Load retrievers for existing files
    for f in current_files.values():
        name = os.path.splitext(f)[0]
        retriever = load_vector_store(f)
        if retriever:
            retrievers[name] = retriever
    return retrievers

if __name__ == "__main__":
    load_all_retrievers()

import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import os
import time

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")

api_key = st.secrets.get("GOOGLE_API_KEY")

class GoogleDirectEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    def embed_documents(self, texts):
        embeddings = []
        progreso = st.progress(0, text="Indexando contenido (Solo la primera vez)...")
        for i, t in enumerate(texts):
            try:
                res = genai.embed_content(model="models/gemini-embedding-001", content=t, task_type="retrieval_document")
                embeddings.append(res["embedding"])
                time.sleep(2.0) # Pausa para no saturar
                progreso.progress((i + 1) / len(texts))
            except Exception as e:
                st.error(f"Error de cuota: {str(e)}. Probablemente debas esperar 24hs.")
                return []
        progreso.empty()
        return embeddings
    def embed_query(self, text):
        return genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_query")["embedding"]

def obtener_retriever(ruta_archivo, api_key):
    indice_local = "faiss_index"
    model_emb = GoogleDirectEmbeddings(api_key=api_key)
    
    # Si ya procesamos la tesis antes, la cargamos desde el disco (gratis y rápido)
    if os.path.exists(indice_local):
        vectorstore = FAISS.load_local(indice_local, model_emb, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Si no existe, la procesamos (esto consume cuota)
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Fragmentos mucho más grandes para ahorrar cuota
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
        chunks = text_splitter.split_text(text)
        
        vectorstore = FAISS.from_texts(chunks, model_emb)
        # GUARDAMOS EL ÍNDICE para no volver a gastar cuota
        vectorstore.save_local(indice_local)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error al inicializar: {str(e)}")
        return None

# 2. Lógica de Chat
if api_key:
    archivo_txt = "tesis_mauricio.txt"
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_txt):
            st.session_state.retriever = obtener_retriever(archivo_txt, api_key)
        else:
            st.error("Falta tesis_mauricio.txt")

    if st.session_state.get("retriever"):
        pregunta = st.text_input("Consultá sobre el microscopio o simulaciones:")
        if pregunta:
            try:
                docs = st.session_state.retriever.invoke(pregunta)
                contexto = "\n\n".join([doc.page_content for doc in docs])
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"Responde como físico experto basándote en este contexto:\n{contexto}\n\nPregunta: {pregunta}"
                
                with st.spinner("Pensando..."):
                    response = model.generate_content(prompt)
                    st.info(response.text)
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY.")

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
        progreso = st.progress(0, text="Indexando contenido técnico...")
        for i, t in enumerate(texts):
            try:
                res = genai.embed_content(model="models/gemini-embedding-001", content=t, task_type="retrieval_document")
                embeddings.append(res["embedding"])
                time.sleep(2.0) # Pausa aumentada para cuidar la cuota diaria
                progreso.progress((i + 1) / len(texts))
            except Exception as e:
                if "429" in str(e):
                    st.warning("Límite diario alcanzado. Esperando un momento para reintentar...")
                    time.sleep(60) # Espera un minuto completo si hay bloqueo
                    res = genai.embed_content(model="models/gemini-embedding-001", content=t, task_type="retrieval_document")
                    embeddings.append(res["embedding"])
                else:
                    raise e
        progreso.empty()
        return embeddings
    def embed_query(self, text):
        return genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_query")["embedding"]

def procesar_texto(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            text = f.read()
        
        # FRAGMENTOS MÁS GRANDES: Clave para no agotar las 1000 peticiones diarias
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
        chunks = text_splitter.split_text(text)
        
        model_emb = GoogleDirectEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, model_emb)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error en procesamiento: {str(e)}")
        return None

# 2. Lógica de Chat
if api_key:
    archivo_txt = "tesis_mauricio.txt"
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_txt):
            with st.spinner("Cargando conocimientos de la tesis..."):
                st.session_state.retriever = procesar_texto(archivo_txt)
        else:
            st.error(f"No se encontró el archivo '{archivo_txt}'.")

    if st.session_state.get("retriever"):
        pregunta = st.text_input("Realizá tu consulta técnica sobre la tesis:")
        if pregunta:
            try:
                retriever = st.session_state.retriever
                docs = retriever.invoke(pregunta)
                contexto = "\n\n".join([doc.page_content for doc in docs])
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""Responde como experto en física basándote SOLO en este contexto:
                {contexto}
                
                Pregunta: {pregunta}"""
                
                with st.spinner("Buscando en la tesis..."):
                    response = model.generate_content(prompt)
                    st.info(response.text)
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY en los Secrets.")

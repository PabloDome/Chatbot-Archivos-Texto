import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import os
import time

# 1. Configuración de la página
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")

api_key = st.secrets.get("GOOGLE_API_KEY")

# Clase corregida para heredar de Embeddings y evitar el error "not callable"
class GoogleDirectEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def embed_documents(self, texts):
        embeddings = []
        progreso = st.progress(0, text="Indexando contenido científico...")
        for i, t in enumerate(texts):
            try:
                res = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=t,
                    task_type="retrieval_document"
                )
                embeddings.append(res["embedding"])
                time.sleep(1.5) # Respetamos cuota
                progreso.progress((i + 1) / len(texts))
            except Exception:
                time.sleep(10)
                res = genai.embed_content(model="models/gemini-embedding-001", content=t, task_type="retrieval_document")
                embeddings.append(res["embedding"])
        progreso.empty()
        return embeddings
    
    def embed_query(self, text):
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]

def procesar_texto(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        # Instanciamos la clase corregida
        embeddings_model = GoogleDirectEmbeddings(api_key=api_key)
        
        # Pasamos el objeto directamente
        vectorstore = FAISS.from_texts(chunks, embeddings_model)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None

# 2. Lógica de carga y Chat
if api_key:
    archivo_txt = "tesis_mauricio.txt"
    
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_txt):
            with st.spinner("Cargando conocimientos de la tesis..."):
                st.session_state.retriever = procesar_texto(archivo_txt)
        else:
            st.error(f"No se encontró el archivo '{archivo_txt}' en el repositorio.")

    if "retriever" in st.session_state and st.session_state.retriever:
        pregunta = st.text_input("Realizá tu consulta técnica:")
        
        if pregunta:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key,
                    temperature=0.2
                )
                
                prompt = ChatPromptTemplate.from_template("""
                Eres un experto en física experimental. 
                Responde basándote únicamente en el contexto de la tesis provisto.
                
                Contexto: {context}
                Pregunta: {question}
                
                Respuesta técnica:""")

                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                
                with st.spinner("Buscando en la tesis..."):
                    st.info(chain.invoke(pregunta))
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY en Secrets.")

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

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")

api_key = st.secrets.get("GOOGLE_API_KEY")

# Clase de Embeddings optimizada para tu cuenta
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
                time.sleep(1.5) 
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
        embeddings_model = GoogleDirectEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings_model)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error al procesar: {str(e)}")
        return None

# 2. Lógica de Chat
if api_key:
    archivo_txt = "tesis_mauricio.txt"
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_txt):
            with st.spinner("Cargando conocimientos..."):
                st.session_state.retriever = procesar_texto(archivo_txt)
        else:
            st.error(f"No se encontró '{archivo_txt}'")

    if st.session_state.get("retriever"):
        pregunta = st.text_input("Realizá tu consulta técnica:")
        if pregunta:
            try:
                # CAMBIO CLAVE: Forzamos el uso de la versión v1 de la API para evitar el 404
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key,
                    temperature=0.2
                )
                
                prompt = ChatPromptTemplate.from_template("""
                Eres un experto en física. Responde basándote en la tesis provista.
                Contexto: {context}
                Pregunta: {question}
                Respuesta:""")

                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                
                with st.spinner("Buscando respuesta..."):
                    st.info(chain.invoke(pregunta))
            except Exception as e:
                # Si sigue dando error de versión, intentamos con el nombre alternativo
                st.warning("Reintentando con configuración de respaldo...")
                try:
                    llm_alt = ChatGoogleGenerativeAI(
                        model="models/gemini-1.5-flash", # Nombre completo con prefijo
                        google_api_key=api_key
                    )
                    chain_alt = ({"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | llm_alt | StrOutputParser())
                    st.info(chain_alt.invoke(pregunta))
                except Exception as e2:
                    st.error(f"Error persistente de API: {str(e2)}")
else:
    st.error("Configurá la GOOGLE_API_KEY.")

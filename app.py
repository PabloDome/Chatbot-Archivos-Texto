import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Consulta técnica optimizada mediante archivo de texto plano.")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_texto(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Al ser texto plano, podemos usar fragmentos más grandes (más contexto, menos llamadas)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        class GoogleDirectEmbeddings:
            def embed_documents(self, texts):
                embeddings = []
                progreso = st.progress(0, text="Indexando contenido técnico...")
                for i, t in enumerate(texts):
                    try:
                        # Pausa de 1.5s para respetar el límite de 100 RPM de la API gratuita
                        res = genai.embed_content(
                            model="models/gemini-embedding-001",
                            content=t,
                            task_type="retrieval_document"
                        )
                        embeddings.append(res["embedding"])
                        time.sleep(1.5)
                        progreso.progress((i + 1) / len(texts))
                    except Exception:
                        st.warning(f"Límite de cuota alcanzado. Reintentando fragmento {i+1}...")
                        time.sleep(10)
                        res = genai.embed_content(model="models/gemini-embedding-001", content=t, task_type="retrieval_document")
                        embeddings.append(res["embedding"])
                progreso.empty()
                return embeddings
            
            def embed_query(self, text):
                return genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type="retrieval_query"
                )["embedding"]

        vectorstore = FAISS.from_texts(chunks, GoogleDirectEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error al procesar el archivo de texto: {str(e)}")
        return None

# 2. Lógica de ejecución
if api_key:
    archivo_txt = "tesis_mauricio.txt"
    
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_txt):
            with st.spinner("Cargando base de conocimientos desde TXT..."):
                st.session_state.retriever = procesar_texto(archivo_txt)
        else:
            st.error(f"❌ No se encontró el archivo '{archivo_txt}' en el repositorio.")
            st.info("Asegúrate de haber subido el archivo de texto que generamos anteriormente.")

    if "retriever" in st.session_state and st.session_state.retriever:
        preg = st.text_input("Realizá una consulta sobre el microscopio o la dinámica de paredes:")
        if preg:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente experto en física. Responde basándote en la tesis: {context}\n\nPregunta: {question}"
                )
                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                st.info(chain.invoke(preg))
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY en los Secrets de Streamlit.")

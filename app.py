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

# 1. Configuración de la página
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Pregunta lo que sea, seguro que no tengo idea.")

# Obtención de la API Key desde Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_texto(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Fragmentación del texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        class GoogleDirectEmbeddings:
            def embed_documents(self, texts):
                embeddings = []
                progreso = st.progress(0, text="Indexando contenido científico...")
                for i, t in enumerate(texts):
                    try:
                        # Pausa de 1.5s para respetar el límite de la API gratuita (100 RPM)
                        res = genai.embed_content(
                            model="models/gemini-embedding-001",
                            content=t,
                            task_type="retrieval_document"
                        )
                        embeddings.append(res["embedding"])
                        time.sleep(1.5)
                        progreso.progress((i + 1) / len(texts))
                    except Exception:
                        # Reintento largo en caso de error de cuota
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

        # Crear base de datos vectorial
        vectorstore = FAISS.from_texts(chunks, GoogleDirectEmbeddings())
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
                # Configuración del modelo de lenguaje
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key,
                    temperature=0.2 # Precisión técnica alta
                )
                
                # Template para asegurar que use el contexto de la tesis
                prompt = ChatPromptTemplate.from_template("""
                Eres un experto en física experimental. 
                Responde la pregunta basándote únicamente en el contexto provisto de la tesis.
                Si la respuesta no está clara, intenta deducirla profesionalmente o indícalo.

                Contexto de la tesis:
                {context}

                Pregunta:
                {question}

                Respuesta técnica:""")

                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt 
                    | llm 
                    | StrOutputParser()
                )
                
                with st.spinner("Buscando en la tesis..."):
                    respuesta = chain.invoke(pregunta)
                    st.info(respuesta)
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Por favor, configura la GOOGLE_API_KEY en los Secrets de Streamlit.")

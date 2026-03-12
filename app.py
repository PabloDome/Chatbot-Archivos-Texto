import streamlit as st
import PyPDF2
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

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Chunks grandes para hacer menos llamadas a la API
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        class GoogleDirectEmbeddings:
            def embed_documents(self, texts):
                embeddings = []
                progreso = st.progress(0, text="Procesando fragmentos (respetando cuota)...")
                for i, t in enumerate(texts):
                    # Pausa de 1.5 segundos para evitar el error 429 de Google
                    try:
                        res = genai.embed_content(
                            model="models/gemini-embedding-001", 
                            content=t, 
                            task_type="retrieval_document"
                        )
                        embeddings.append(res["embedding"])
                        time.sleep(1.5) 
                        progreso.progress((i + 1) / len(texts))
                    except Exception as e:
                        st.warning(f"Pausa técnica por límite de cuota... reintentando fragmento {i+1}")
                        time.sleep(10) # Espera larga si hay error
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
        st.error(f"No se pudo procesar el PDF: {str(e)}")
        return None

# 2. Lógica de ejecución
if api_key:
    archivo_pdf = "tesis_mauricio.pdf"
    
    if "retriever" not in st.session_state:
        if os.path.exists(archivo_pdf):
            with st.spinner("Analizando la tesis..."):
                resultado = procesar_pdf(archivo_pdf)
                if resultado:
                    st.session_state.retriever = resultado
                    st.success("Tesis cargada correctamente.")
        else:
            st.warning("Subí el PDF para comenzar:")
            u_file = st.file_uploader("Archivo PDF", type="pdf")
            if u_file:
                st.session_state.retriever = procesar_pdf(u_file)

    if "retriever" in st.session_state and st.session_state.retriever:
        preg = st.text_input("Hacé tu pregunta técnica:")
        if preg:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template("Responde según la tesis: {context}\n\nPregunta: {question}")
                chain = ({"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                st.info(chain.invoke(preg))
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY en Secrets.")

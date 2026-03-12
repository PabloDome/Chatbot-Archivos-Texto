import os
# PARCHE DE EMERGENCIA: Forzamos la versión estable antes de cualquier import
os.environ["GOOGLE_API_VERSION"] = "v1"

import streamlit as st
import PyPDF2
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Interfaz y Configuración
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        
        genai.configure(api_key=api_key)
        
        # Clase personalizada usando explícitamente la API estable
        class GoogleStableEmbeddings:
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"] for t in texts]
            def embed_query(self, text):
                return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")["embedding"]

        vectorstore = FAISS.from_texts(chunks, GoogleStableEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")
        return None

# 2. Lógica de Aplicación
if api_key:
    nombre_archivo = "tesis_mauricio.pdf"
    
    if "retriever" not in st.session_state:
        # Buscamos el archivo en el repositorio
        if os.path.exists(nombre_archivo):
            with st.spinner("Procesando tesis..."):
                st.session_state.retriever = procesar_pdf(nombre_archivo)
        else:
            st.warning("No se encontró el archivo. Podés subirlo manualmente:")
            u_file = st.file_uploader("Subir PDF", type="pdf")
            if u_file:
                st.session_state.retriever = procesar_pdf(u_file)

    if "retriever" in st.session_state and st.session_state.retriever:
        preg = st.text_input("Consultá detalles sobre el microscopio o simulaciones:")
        if preg:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            prompt = ChatPromptTemplate.from_template("Responde según la tesis: {context}\n\nPregunta: {question}")
            chain = ({"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
            st.info(chain.invoke(preg))
else:
    st.error("Configurá la GOOGLE_API_KEY en los Secrets.")

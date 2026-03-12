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

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        # Lectura y fragmentación
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        
        # CONFIGURACIÓN DE TRANSPORTE: Forzamos el uso de la API estable
        genai.configure(api_key=api_key)
        
        class GoogleStableEmbeddings:
            def embed_documents(self, texts):
                # Usamos el modelo 004 con la configuración de tarea específica
                return [genai.embed_content(
                    model="models/text-embedding-004", 
                    content=t, 
                    task_type="retrieval_document"
                )["embedding"] for t in texts]
                
            def embed_query(self, text):
                return genai.embed_content(
                    model="models/text-embedding-004", 
                    content=text, 
                    task_type="retrieval_query"
                )["embedding"]

        # Creamos el almacén vectorial
        vectorstore = FAISS.from_texts(chunks, GoogleStableEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    except Exception as e:
        # Si falla, imprimimos el error exacto para diagnosticar
        st.error(f"Error detallado de la API: {str(e)}")
        return None

# 2. Lógica de ejecución
if api_key:
    nombre_archivo = "tesis_mauricio.pdf"
    
    if "retriever" not in st.session_state:
        if os.path.exists(nombre_archivo):
            with st.spinner("Procesando tesis con Google API v1..."):
                st.session_state.retriever = procesar_pdf(nombre_archivo)
        else:
            st.warning("⚠️ No se encontró el PDF en GitHub.")
            u_file = st.file_uploader("Subir PDF manualmente", type="pdf")
            if u_file:
                st.session_state.retriever = procesar_pdf(u_file)

    if "retriever" in st.session_state and st.session_state.retriever:
        pregunta = st.text_input("Consultá detalles sobre el microscopio o simulaciones:")
        if pregunta:
            try:
                # Inicialización del chat usando Gemini 1.5 Flash
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template("Responde basándote en la tesis: {context}\n\nPregunta: {question}")
                chain = ({"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                st.info(chain.invoke(pregunta))
            except Exception as e:
                st.error(f"Error en Gemini: {str(e)}")
else:
    st.error("Falta la API Key en los Secrets de Streamlit.")

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
st.markdown("Consultá detalles técnicos sobre el microscopio y el efecto Kerr.")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Chunks más grandes para reducir la cantidad de peticiones a la API
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        class GoogleDirectEmbeddings:
            def embed_documents(self, texts):
                embeddings = []
                barra_progreso = st.progress(0)
                total = len(texts)
                
                for i, t in enumerate(texts):
                    # Pausa de 1 segundo para evitar el error 429 (Quota exceeded)
                    res = genai.embed_content(
                        model="models/gemini-embedding-001", 
                        content=t, 
                        task_type="retrieval_document"
                    )
                    embeddings.append(res["embedding"])
                    time.sleep(1) 
                    barra_progreso.progress((i + 1) / total)
                return embeddings
                
            def embed_query(self, text):
                return genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=text, 
                    task_type="retrieval_query"
                )["embedding"]

        vectorstore = FAISS.from_texts(chunks, GoogleDirectEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    except Exception as e:
        st.error(f"Error técnico o de cuota: {str(e)}")
        return None

# 2. Lógica de carga y chat
if api_key:
    nombre_archivo = "tesis_mauricio.pdf"
    
    if "retriever" not in st.session_state:
        if os.path.exists(nombre_archivo):
            with st.spinner("Analizando la tesis (esto tomará unos minutos por los límites de cuota)..."):
                st.session_state.retriever = procesar_pdf(nombre_archivo)
        else:
            st.warning("⚠️ No se encontró el PDF en GitHub. Subilo manualmente:")
            u_file = st.file_uploader("Subir PDF", type="pdf")
            if u_file:
                st.session_state.retriever = procesar_pdf(u_file)

    if "retriever" in st.session_state and st.session_state.retriever:
        pregunta = st.text_input("Consultá detalles sobre el microscopio o simulaciones:")
        if pregunta:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template("Responde según la tesis: {context}\n\nPregunta: {question}")
                chain = ({"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                st.info(chain.invoke(pregunta))
            except Exception as e:
                st.error(f"Error en Gemini: {str(e)}")
else:
    st.error("Configurá la GOOGLE_API_KEY en los Secrets de Streamlit.")

import streamlit as st
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import os

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Consultá detalles técnicos sobre el microscopio y el efecto Kerr.")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        # Clase para evitar el error 404 de la v1beta en LangChain
        class GoogleCustomEmbeddings:
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"] for t in texts]
            def embed_query(self, text):
                return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")["embedding"]

        vectorstore = FAISS.from_texts(chunks, GoogleCustomEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error técnico al procesar: {str(e)}")
        return None

# 2. Lógica de Carga de Archivo
if api_key:
    # Intento 1: Cargar desde el repositorio automáticamente
    nombre_archivo = "tesis_mauricio.pdf"
    if not os.path.exists(nombre_archivo):
        # Si falla el nombre estándar, buscamos en el directorio actual
        base_path = os.path.dirname(__file__)
        nombre_archivo = os.path.join(base_path, "tesis_mauricio.pdf")

    if "retriever" not in st.session_state:
        if os.path.exists(nombre_archivo):
            with st.spinner("Analizando archivo del repositorio..."):
                st.session_state.retriever = procesar_pdf(nombre_archivo)
        else:
            st.warning("⚠️ No se encontró el PDF en el repositorio. Por favor, subilo manualmente abajo.")
            uploaded_file = st.file_uploader("Subir tesis_mauricio.pdf", type="pdf")
            if uploaded_file:
                st.session_state.retriever = procesar_pdf(uploaded_file)

    # 3. Chatbot
    if "retriever" in st.session_state and st.session_state.retriever:
        pregunta = st.text_input("Hacé tu pregunta sobre la tesis:")
        
        if pregunta:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente experto en física. Respondé basándote en este contexto de la tesis:\n{context}\n\nPregunta: {question}"
                )
                
                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                
                with st.spinner("Buscando en la tesis..."):
                    st.info(chain.invoke(pregunta))
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Falta GOOGLE_API_KEY en los Secrets de Streamlit.")import streamlit as st
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import os

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Consultá detalles técnicos sobre el microscopio y el efecto Kerr.")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        genai.configure(api_key=api_key)
        
        # Clase para evitar el error 404 de la v1beta en LangChain
        class GoogleCustomEmbeddings:
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"] for t in texts]
            def embed_query(self, text):
                return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")["embedding"]

        vectorstore = FAISS.from_texts(chunks, GoogleCustomEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error técnico al procesar: {str(e)}")
        return None

# 2. Lógica de Carga de Archivo
if api_key:
    # Intento 1: Cargar desde el repositorio automáticamente
    nombre_archivo = "tesis_mauricio.pdf"
    if not os.path.exists(nombre_archivo):
        # Si falla el nombre estándar, buscamos en el directorio actual
        base_path = os.path.dirname(__file__)
        nombre_archivo = os.path.join(base_path, "tesis_mauricio.pdf")

    if "retriever" not in st.session_state:
        if os.path.exists(nombre_archivo):
            with st.spinner("Analizando archivo del repositorio..."):
                st.session_state.retriever = procesar_pdf(nombre_archivo)
        else:
            st.warning("⚠️ No se encontró el PDF en el repositorio. Por favor, subilo manualmente abajo.")
            uploaded_file = st.file_uploader("Subir tesis_mauricio.pdf", type="pdf")
            if uploaded_file:
                st.session_state.retriever = procesar_pdf(uploaded_file)

    # 3. Chatbot
    if "retriever" in st.session_state and st.session_state.retriever:
        pregunta = st.text_input("Hacé tu pregunta sobre la tesis:")
        
        if pregunta:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente experto en física. Respondé basándote en este contexto de la tesis:\n{context}\n\nPregunta: {question}"
                )
                
                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                
                with st.spinner("Buscando en la tesis..."):
                    st.info(chain.invoke(pregunta))
            except Exception as e:
                st.error(f"Error en la consulta: {str(e)}")
else:
    st.error("Falta GOOGLE_API_KEY en los Secrets de Streamlit.")

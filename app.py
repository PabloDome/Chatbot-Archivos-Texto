import streamlit as st
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Consultá detalles técnicos sobre el microscopio y el efecto Kerr.")

# Obtención de la API Key desde Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_documento():
    # Buscamos específicamente el archivo confirmado
    nombre_archivo = "tesis_mauricio.pdf"
    
    if not os.path.exists(nombre_archivo):
        st.error(f"Error: No se encontró el archivo '{nombre_archivo}' en el repositorio.")
        return None
    
    try:
        pdf_reader = PyPDF2.PdfReader(nombre_archivo)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        # Usamos el modelo estable embedding-001
        embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=api_key,
    client_options={"api_version": "v1"} # Esto fuerza la versión estable
)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error al procesar el documento: {str(e)}")
        return None

# 2. Lógica del Chatbot
if api_key:
    if "retriever" not in st.session_state:
        with st.spinner("Analizando la tesis..."):
            st.session_state.retriever = procesar_documento()
    
    retriever = st.session_state.retriever
    
    if retriever:
        pregunta = st.text_input("Hacé tu pregunta técnica:")
        
        if pregunta:
            try:
                # Se agregó la importación correcta de ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                
                template = """Respondé la pregunta basándote solo en el siguiente contexto técnico de la tesis:
                {context}
                
                Pregunta: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
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
    st.error("Falta GOOGLE_API_KEY en Secrets de Streamlit.")

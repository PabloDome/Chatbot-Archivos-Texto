import streamlit as st
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("Consultá detalles técnicos sobre el microscopio y el efecto Kerr.")

api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_documento():
    nombre_archivo = "Tesis-Mauricio.pdf"
    try:
        pdf_reader = PyPDF2.PdfReader(nombre_archivo)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error: Aseguráte de que '{nombre_archivo}' esté en GitHub.")
        return None

# 2. Lógica del Chatbot (Versión 2026 estable)
if api_key:
    if "retriever" not in st.session_state:
        with st.spinner("Analizando la tesis..."):
            st.session_state.retriever = procesar_documento()
    
    retriever = st.session_state.retriever
    
    if retriever:
        pregunta = st.text_input("Hacé tu pregunta técnica:")
        
        if pregunta:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                
                # Definimos cómo debe responder la IA
                template = """Respondé la pregunta basándote solo en el siguiente contexto técnico de la tesis:
                {context}
                
                Pregunta: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                # Esta cadena de comandos reemplaza al viejo RetrievalQA que fallaba
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
                st.error("Error en la consulta. Verificá la clave en Secrets.")
else:
    st.error("Falta GOOGLE_API_KEY en Secrets.")

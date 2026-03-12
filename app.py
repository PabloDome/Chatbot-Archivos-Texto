import streamlit as st
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Configuración de la interfaz
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("""
Consultá detalles técnicos sobre la tesis: 
**'Construcción de un Microscopio Magneto-Óptico para el estudio de películas Ultradelgadas'**.
""")

# 2. Conexión automática con la clave de Google (Invisible para el usuario)
# Busca la clave 'GOOGLE_API_KEY' en la configuración interna de Streamlit
api_key = st.secrets.get("GOOGLE_API_KEY")

def procesar_documento():
    # El archivo debe tener este nombre exacto en tu repositorio de GitHub
    nombre_archivo = "tesis-romano (1).pdf"
    try:
        pdf_reader = PyPDF2.PdfReader(nombre_archivo)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Dividimos el texto para que la IA localice mejor los datos técnicos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        # Creamos la base de datos de conocimientos con embeddings de Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error al cargar el PDF: Aseguráte de que el archivo se llama '{nombre_archivo}' en tu GitHub.")
        return None

# 3. Lógica del Chatbot
if api_key:
    # Usamos la memoria de sesión para no re-analizar el PDF en cada pregunta
    if "vector_db" not in st.session_state:
        with st.spinner("Analizando contenido técnico de la tesis..."):
            st.session_state.vector_db = procesar_documento()
    
    db = st.session_state.vector_db
    
    if db:
        pregunta = st.text_input("Hacé una pregunta sobre el microscopio, el software o el efecto Kerr:")
        
        if pregunta:
            try:
                # Usamos Gemini 1.5 Flash para respuestas rápidas y precisas
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=db.as_retriever(search_kwargs={"k": 5})
                )
                
                with st.spinner("Buscando en la tesis..."):
                    respuesta = qa_chain.invoke(pregunta)
                    st.info(respuesta["result"])
            except Exception as e:
                st.error("Hubo un error en la consulta. Verificá si la clave en Secrets es correcta.")
else:
    # Este mensaje solo se verá si falta la clave en el panel de Streamlit Cloud
    st.error("⚠️ Error de configuración: No se encontró la clave 'GOOGLE_API_KEY' en los Secrets.")
    st.info("Para solucionar esto, andá a Settings -> Secrets en tu panel de Streamlit y agregá la clave.")

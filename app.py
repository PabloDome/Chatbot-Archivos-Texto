import streamlit as st
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Configuración de la página
st.set_page_config(page_title="Asistente de Tesis - M. E. Romano", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de Mauricio Romano")
st.markdown("""
Esta IA ha sido entrenada con el contenido de la tesis **"Construcción de un Microscopio Magneto-Óptico para el estudio de películas Ultradelgadas"**.
""")

# Gestión de la API Key (Busca en Secrets o pide manual)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Introduce tu Google API Key", type="password")
    st.sidebar.info("Para que este chat funcione, necesitas una clave de Google AI Studio.")

def procesar_documento():
    # 1. Leer el PDF que ya está en tu repositorio
    nombre_archivo = "tesis-romano (1).pdf"
    try:
        pdf_reader = PyPDF2.PdfReader(nombre_archivo)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 2. Dividir el texto en fragmentos manejables
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        
        # 3. Crear el motor de búsqueda (Embeddings)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {nombre_archivo} en el repositorio.")
        return None

# Ejecución principal
if api_key:
    try:
        # Usamos cache para que no analice el PDF cada vez que haces una pregunta
        if "vector_db" not in st.session_state:
            with st.spinner("Analizando la tesis... Esto toma unos segundos."):
                st.session_state.vector_db = procesar_documento()
        
        db = st.session_state.vector_db
        
        if db:
            pregunta = st.text_input("Haz una pregunta técnica (ej: ¿Cómo es el diseño de las bobinas? o ¿Qué es el efecto Kerr?)")
            
            if pregunta:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=db.as_retriever(search_kwargs={"k": 5})
                )
                
                with st.spinner("Consultando la tesis..."):
                    respuesta = qa_chain.invoke(pregunta)
                    st.info(respuesta["result"])
                    
    except Exception as e:
        st.error(f"Ocurrió un error de conexión. Verifica tu API Key.")
else:
    st.warning("👈 Por favor, ingresa tu Google API Key en la barra lateral para comenzar.")

import streamlit as st
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asistente de Tesis - Microscopía MO", page_icon="🔬")

st.title("🔬 Asistente Virtual: Tesis de M. E. Romano")
st.markdown("Consulta detalles técnicos sobre la construcción del microscopio magneto-óptico (Efecto Kerr).")

# Barra lateral para la configuración
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.info("Esta app analiza el PDF de la tesis de Mauricio Romano utilizando IA.")

def cargar_datos():
    # Lee el PDF que ya subiste al repositorio
    pdf_reader = PyPDF2.PdfReader("tesis-romano (1).pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Divide el texto en fragmentos para procesarlo mejor
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

if api_key:
    try:
        with st.spinner("Analizando la tesis..."):
            db = cargar_datos()
        
        pregunta = st.text_input("¿Qué quieres saber sobre el microscopio, el montaje o los resultados?")
        
        if pregunta:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
            respuesta = qa.invoke(pregunta)
            st.info(respuesta["result"])
            
    except Exception as e:
        st.error(f"Error: {e}. Revisa si tu API Key es válida.")
else:
    st.warning("Introduce tu OpenAI API Key en el menú lateral para empezar.")

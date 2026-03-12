import streamlit as st
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asistente Tesis Romano", page_icon="🔬")
st.title("🔬 Asistente Virtual: Tesis M. E. Romano")

# Aquí pides la clave de Google (es gratis)
api_key = st.sidebar.text_input("Introduce tu Google API Key", type="password")
st.sidebar.markdown("[Consigue tu clave gratis aquí](https://aistudio.google.com/app/apikey)")

def procesar_tesis():
    pdf_reader = PyPDF2.PdfReader("tesis-romano (1).pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

if api_key:
    try:
        with st.spinner("Analizando la tesis de Mauricio..."):
            db = procesar_tesis()
        
        pregunta = st.text_input("Pregunta sobre el microscopio, el software o la binarización:")
        if pregunta:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
            respuesta = qa.invoke(pregunta)
            st.info(respuesta["result"])
    except Exception as e:
        st.error(f"Error de conexión: Asegúrate de que la clave sea de Google Gemini.")
else:
    st.warning("Por favor, introduce la Google API Key en la barra lateral.")

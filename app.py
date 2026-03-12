import google.generativeai as genai

def procesar_pdf(pdf_file):
    try:
        # 1. Lectura
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        
        # 2. Configuración del SDK (Forzando la ruta estable)
        genai.configure(api_key=api_key)
        
        class GoogleStableEmbeddings:
            def embed_documents(self, texts):
                # Usamos text-embedding-004 que es el estándar actual en v1
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

        # 3. Creación del índice
        vectorstore = FAISS.from_texts(chunks, GoogleStableEmbeddings())
        return vectorstore.as_retriever(search_kwargs={"k": 5})
        
    except Exception as e:
        # Si esto falla, el log nos dirá exactamente qué versión intentó usar
        st.error(f"Error detallado: {str(e)}")
        return None

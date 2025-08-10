import os
import tempfile
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Fix event loop issue for Google AI
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_pdf_db")

def initialize_vector_store(pdf_file, api_key):
    """Initialize vector store with PDF content"""
    # Create temporary file for PDF processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        temp_pdf_path = f.name
    
    try:
        # Load PDF document
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Daha büyük chunk boyutu
            chunk_overlap=400  # Daha fazla overlap
        )
        docs = text_splitter.split_documents(documents)
        
        # Add metadata to documents
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Create and persist vector store
        db = Chroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=persistent_directory
        )
        
        return db, len(docs)
        
    finally:
        # Clean up temporary file
        os.remove(temp_pdf_path)

def load_existing_vector_store(api_key):
    """Load existing vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

def get_full_document_summary(db, api_key):
    """PDF'in tamamından özet çıkar"""
    try:
        # Tüm chunk'ları al
        all_docs = db.get()['documents']
        combined_text = " ".join(all_docs)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        prompt = f"""Bu PDF dökümanının tam metnini analiz et ve detaylı bir Türkçe özet çıkar:

{combined_text}

Lütfen:
1. Ana konuları belirt
2. Önemli detayları dahil et  
3. Planın tüm bölümlerini kaps
4. Hedefleri ve stratejileri açıkla
5. Kapsamlı ve ayrıntılı bir özet hazırla"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content
    except Exception as e:
        return f"Özet çıkarılırken hata oluştu: {str(e)}"

def create_rag_chain(db, api_key):
    """Create RAG chain for conversational PDF chat"""
    # Create retriever - özet soruları için tüm chunk'ları getir
    total_chunks = db._collection.count()
    k_value = min(total_chunks, 20) if total_chunks > 0 else 20
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_value}  # Mümkün olduğunca fazla chunk
    )
    
    # Create ChatGoogleGenerativeAI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer question prompt
    qa_system_prompt = (
        "Sen PDF dökümanları hakkında soruları yanıtlayan bir asistansın. "
        "Aşağıdaki bağlamı kullanarak soruları yanıtla. "
        "Türkçe cevap ver. PDF'in tamamını dikkate alarak kapsamlı analiz yap. "
        "Eğer özet isteniyorsa, tüm önemli noktaları içeren detaylı bir özet hazırla. "
        "Eğer PDF'in ne hakkında olduğu soruluyorsa, ana konuları ve içeriği açıkla. "
        "Sadece verilen bağlamı kullan ve bilmediğin bir şey varsa belirt."
        "\n\n"
        "Bağlam: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Streamlit UI
st.title("Chat with PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Google API Key input
google_api_key = st.text_input("Google API Key", type="password", help="Get your key from https://aistudio.google.com/apikey")

if google_api_key:
    # PDF file uploader
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            try:
                # Initialize vector store with PDF
                db, num_chunks = initialize_vector_store(pdf_file, google_api_key)
                
                # Create RAG chain
                st.session_state.rag_chain = create_rag_chain(db, google_api_key)
                st.session_state.pdf_processed = True
                
                st.success(f"Added {pdf_file.name} to knowledge base! ({num_chunks} chunks created)")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    elif os.path.exists(persistent_directory) and st.session_state.rag_chain is None:
        # Load existing vector store if available
        with st.spinner("Loading existing knowledge base..."):
            try:
                db = load_existing_vector_store(google_api_key)
                st.session_state.rag_chain = create_rag_chain(db, google_api_key)
                st.info("Loaded existing knowledge base.")
            except Exception as e:
                st.error(f"Error loading existing knowledge base: {str(e)}")
    
    # Özet butonu
    if st.session_state.rag_chain:
        if st.button("📄 PDF'in Tam Özetini Çıkar"):
            with st.spinner("PDF'in tamamı analiz ediliyor..."):
                try:
                    db = load_existing_vector_store(google_api_key)
                    summary = get_full_document_summary(db, google_api_key)
                    st.success("PDF Özeti:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Özet çıkarılamadı: {str(e)}")
    
    # Chat interface
    if st.session_state.rag_chain:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, SystemMessage):
                st.chat_message("assistant").write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF"):
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = result["answer"]
                    
                    # Display AI response
                    st.chat_message("assistant").write(answer)
                    
                    # Update chat history
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(SystemMessage(content=answer))
                    
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
    
    else:
        st.info("Please upload a PDF file to start chatting.")

else:
    st.info("Please enter your Google API Key from https://aistudio.google.com/apikey to start.")

# Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
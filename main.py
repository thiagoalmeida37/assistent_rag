import os
import tempfile

import streamlit as st

from decouple import config

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

persist_directory = 'db'

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    os.remove(temp_file_path)  # Clean up the temporary file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function = OpenAIEmbeddings(),
        )
    
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

vector_store = load_existing_vector_store()


st.set_page_config(
    page_title="Assistente Virtual",
    page_icon=":robot:",
)

st.title('Assistente Virtual')

st.header('🤖 Bem-vindo ao Assistente Virtual com RAG')

with st.sidebar:
    st.header('Upload de Arquivos 🧾')
    uploaded_files = st.file_uploader(
        label='Carregue seus arquivos aqui',
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner('Processando arquivos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
                print(all_chunks)
            
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store = vector_store,
            )

            st.success('Arquivos processados com sucesso!')


    st.header('Configurações do Modelo LLM 🛠️')
    model_options = {
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    }

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
        index=0,
    )
    
question = st.chat_input('Faça uma pergunta ao Assistente Virtual')
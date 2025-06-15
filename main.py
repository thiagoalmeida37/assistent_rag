import os
import tempfile

import streamlit as st

from decouple import config

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

persist_directory = 'db'

def process_pdf(file):
    """
    Processa o arquivo PDF recebido como parâmetro.

    Esta função normalmente realiza operações como:
    - Abrir o arquivo PDF.
    - Ler e extrair o texto das páginas.
    - Retornar o texto extraído ou processado para uso posterior.

    Parâmetros:
        file: Caminho ou objeto do arquivo PDF a ser processado.

    Retorna:
        O texto extraído do PDF ou outro resultado do processamento.
    """
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
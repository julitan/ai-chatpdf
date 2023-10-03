__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="harkjael", floating=True, width=221)

# title
st.title("MyChatPDF")
st.write("---")

# openAI Key 입력 받기
openai_key = st.text_input('OPEN_AI_INPUT_KEY', type='password')


# file upload
uploaded_file = st.file_uploader("Choose a pdf file", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:    
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드되면 동작되는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    # Embeddings
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)


    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("myChatPDF에게 질문!")
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기"):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result["result"])

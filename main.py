import streamlit as st
from streamlit_chat import message
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from keys import OPENAI_API_KEY
import os

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


def get_vectorstore(software):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = FAISS.load_local(f'/home/yuvraj/projects/docai/vector_stores/{software}', embeddings, index_name=software)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput():
    response = st.session_state.conversation({'question': st.session_state.input})
    st.session_state.chat_history = response['chat_history']

    # response_container = st.container()
    # with response_container:
    #     for i, msg in enumerate(st.session_state.chat_history):
    #         if i % 2 == 0:
    #             message(msg.content, is_user=True, key=str(i)+'_user')
    #         else:
    #             message(msg.content, key=str(i))



def main():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    st.set_page_config(page_title="Chat with different documentations", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with different documentations :books:")
    response_container = st.container()
    with response_container:
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i)+'_user')
                else:
                    message(msg.content, key=str(i))

    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
        }}
    </style>
    """
    st.markdown(styl, unsafe_allow_html=True)
    user_question = st.text_input("Ask a question:", key="input", on_change=handle_userinput)
    # if user_question:
    #     handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Available Documentations")
        software = st.selectbox('Which software documentation you want to chat with', ('', 'ansible'))
        if st.button("New Conversation"):
            with st.spinner("Processing"):
                # get pdf text
                # raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                # text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(software)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
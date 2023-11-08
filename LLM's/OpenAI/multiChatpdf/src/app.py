# Import Libraries
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css, bot_template, user_template


st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

#  ---

# hide hamburger and customize footer
hide_menu= """
<style>

#MainMenu {
    visibility:hidden;
}

footer{
    visibility:visible;
}

footer:after{
    content: 'With 🫶️ from Shubham Shankar.';
    display:block;
    position:relative;
    color:grey;
    padding;5px;
    top:3px;
}
</style>

"""
# Styling ----------------------------------------------------------------------

st.image("/Users/shubhamrathod/PycharmProjects/multiChatpdf/pictures/icon.jpg", width=85)
st.title("Chat with Multiple PDF :books:")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """

    Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to **:green[Multi Chat PDF]**! :rocket: This program makes use of **:blue[Large Language Model]** and **:orange[Langchain]** , 
    to chat with document's in **:violet[real-time]** .  ✨

    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:green[Upload File]**: On the side bar, upload `n` number PDF file using the browse button.

    2] **:violet[Processing]**: Click on process for document parsing.
    
    3] **:red[Chat]**: Now type in your question and press enter.

    """
)

st.markdown('---')

st.error(
    """
    Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
    """,
    icon="🧟‍♂️",
)

st.markdown('---')

# ---

# Todo Step 1: Get the text
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text



# Todo Step 2: Break text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    return chunks

# Todo Step 3: Create Vector Store
def get_vectorStore(chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts = chunks, embedding =embeddings)

    return vectorstore


# Todo  Step 4: Create Conversation Chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Todo: Handling User Question
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    # CSS Should always be on top
    st.write(css, unsafe_allow_html=True)

    # Initializing session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
   


    user_question = st.text_input("Ask a question about your documents: ")
    # since we are using embedding  - we donot want to make unnecessary calls
    if user_question:
        handle_userinput(user_question)

    # process documents
    with st.sidebar:
        st.subheader("Your Documents: ")

        pdf_datafile = st.file_uploader("Upload your PDF documents here and click Process", type=['pdf'], accept_multiple_files=True)

        if st.button('Process'):
            if pdf_datafile is not None:
                with st.spinner("Processing"):
                    # Step 1: Get the Text
                    raw_text = get_pdf_text(pdf_datafile)
                    # st.write(raw_text)

                    # Step 2: Get the Chunks
                    text_chunks = get_text_chunks(raw_text)
                    # st.write(text_chunks)
                    # st.write(len(text_chunks))

                    # Step 3: Create Vector Store
                    vectorStore = get_vectorStore(text_chunks)

                    # Step 4: Create Conversation Chain
                    st.session_state.conversation = get_conversation_chain(vectorStore)


if __name__ == '__main__':
    load_dotenv()  # take environment variables from .env.
    main()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64 # display pdf on streamlit
import streamlit as st
import os
import time


# streamlit code for viewing document
st.set_page_config(layout="wide")


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

st.image("icon.jpg", width=85)
st.title("Document Summarization")
st.subheader("LaMini - T5 -> CPU")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """

    Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to my **:green[Document Summarization Application]**! :rocket: This program makes use of **:blue[Large Language Model]** such as **:orange[LaMini]** model, 
    to generate summary of a document using **:violet[CPU]** .  ✨

    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:green[Upload File]**: Upload a PDF file using the provided button.

    2] **:violet[Summary]**: A summarized text of the document is generated.

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






# Todo: Load Model and Tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)


# Todo: File Processing
def file_preProcessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    # print(texts)

    combined_text = ""
    for text in texts:
        combined_text += text.page_content

    # print(combined_text)
    return combined_text

# file_preProcessing('data/invoice.pdf')

# Todo: LLM Pipeline
def LLM_pipeline(file_path):
    # Pipeline
    pipe = pipeline(
        "summarization",
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        min_length = 100,
        early_stopping = True

    )

    input_file = file_preProcessing(file_path)
    print(len(input_file))

    s_result = pipe(input_file)
    # print(s_result)

    summarized_result = s_result[0]['summary_text']

    return summarized_result

# LLM_pipeline('data/invoice.pdf')


# Todo: Display PDF
@st.cache_data
def displayPDF(file_path):
    # Opening file from file path
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)




# Todo: Save Uploaded file
def save_uploadedfile(uploadedfile):
    with open(os.path.join("data/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())



def main():
    st.title("Document Summarization App using Large Language Model")

    datafile = st.file_uploader("Upload file: ", type=['PDF'])

    if datafile is not None:
        file_details = {"FileName": datafile.name, "FileType": datafile.type}
        st.success(file_details)
        save_uploadedfile(datafile)

        if st.button('Summarize'):
            col1 , col2 = st.columns(2)

            file_path = 'data/' + datafile.name

            with open(file_path, "wb") as f:
                f.write(datafile.read())

            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(file_path)

            with col2:
                summarized_result = LLM_pipeline(file_path)
                st.info("Summarized Text")
                with st.spinner('Wait for it...'):
                    time.sleep(5)
                    st.success(summarized_result)




if __name__ == '__main__':
    main()
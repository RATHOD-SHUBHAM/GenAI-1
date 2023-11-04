from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
import streamlit as st

st.title('A :red[_ChatBot_] Welcomes You :sunglasses:')
# st.title('A ChatGPT powered NHTSA issued database')

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index


def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    st.text_area('Here is a example of few questions that you can ask the bot based on the user '
                 'interviews:', '''
                What are some common issues that customers face with their cars?
                How can I identify if my car's battery is dead?
                What should I do if my car's engine overheats?
                Can you tell me more about the importance of regular car maintenance?
                How can I check the oil level in my car?
                What should I do if my car is making strange noises?
                How often should I replace my car's tires?
                What is the typical lifespan of a car's brake pads?
                How can I prevent my car's air conditioning system from failing?
                What is the recommended tire pressure for my car, and how can I check it?
                ''')

    question = st.text_input('What do you want to ask?: ')
    if st.button('Generate'):
        query = question
        response = index.query(query, response_mode="compact")
        print(response.response)
        st.write(response.response)


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "Your API KEY"

    construct_index("/Users/shubhamrathod/PycharmProjects/chatbot/data")

    ask_ai()

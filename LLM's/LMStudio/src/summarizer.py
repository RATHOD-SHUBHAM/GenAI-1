import os
import openai
from langchain.document_loaders import PyPDFLoader

openai.api_base = "http://localhost:1234/v1" # point to the local server
openai.api_key = "" # no need for an API key



loader = PyPDFLoader("tp.pdf")
pages = loader.load_and_split()

summary_list = []
for page in pages:
    # print(page.page_content)
    text = page.page_content
    # print(text)

    text = text + "."


    completion = openai.ChatCompletion.create(
        model='llama2', # this field is currently unused
        messages=[
            {"role": "assistant", "content": "Summarize the text."},
            {"role": "user", "content": text}
        ]
    )

    print(completion.choices[0].message['content'])


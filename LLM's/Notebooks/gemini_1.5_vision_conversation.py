import os
import time
import datetime
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

genai.configure(api_key="google-api-key")
if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "google-api-key"


def upload_to_gemini(path):
  """
    Uploads the given file to Gemini.
  """
  file = genai.upload_file(path)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(*files):
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")


def initializingModel():
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }


    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    ]

    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
    )
    return model

def vision(model,media):
    chat_session = model.start_chat(
    history=[
    {
      "role": "user",
      "parts": [
                    media,
                ],
            },
        ]
    )
    default_prompt = """Tell me a detailed description of what you see in this provided media. 
    Dont title media as an image or video, just say it as "I see". 
    Include every little and make list of dynamic objects, signs, important things with their attirbutes.
    Also provide a summary at end."""

    response = chat_session.send_message(default_prompt)

    print(response.text)
    return response.text

def LLM_call(context,query):
    template = '''
            You are a skilled conversational assistant, and the user is asking you questions while sitting in his car.
            You are witty but keep your tone according to the user query.

            You know the date and time based on the information provided below. Use time and date to answer the queries wherever needed.
            date: {date}

            You also keep track of previous conversations using the chat history.
            Chat History: {history}

            Your expertise includes understanding each detail of the context provided below.
            Context: {response}

            Your Task:
            Understand the user question given below.

            User Question: {input}

            Then, respond appropriately to the user's question while giving suggestions based on the context.
            If you can't make an answer from context then clearly state it to the user "I can't answer this" or similar response.
            While responding, keep your responses short and clear, and use Yes, No, Sure, wherever necessary.
        '''
    prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template,
            partial_variables={ "response": context, "date" : """Current Time:"""+str(datetime.datetime.now().time())+
                            """Current Date"""+str(datetime.date.today())}
        )
    llm= ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=False
        )
    query = query
    response = chain.invoke(query)
    print(response["response"])
    return response["response"]
   

if __name__=="__main__":
   file=r"Screenshot 2024-04-12 181421.png"
   uploaded_file=upload_to_gemini(file)
#    wait_for_files_active(uploaded_file)
   model=initializingModel()
   context=vision(model=model,media=uploaded_file)
   memory = ConversationBufferMemory(memory_key="history", return_messages=True)
   while True:
      user_query=input("User:")
      LLM_call(context=context,query=str(user_query))
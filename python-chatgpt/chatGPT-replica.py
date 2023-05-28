import requests
import streamlit as st

st.title("This is a replica of the OG - chatGPT")

api_url = 'https://api.openai.com/v1/completions'

# Add your api key form setting
api_key = ###

# there is a space after bearer
request_headers = {
    "Content-Type": "application/json" ,
    "Authorization" : "Bearer " + api_key
}


# user_input = input("Ask your question: ")
user_input =  st.text_input("Enter your question 👇")
print(user_input)

if st.button('RUN'):
    request_data = {
        "model": "text-davinci-003",
        "prompt": user_input,
        "max_tokens": 100,
        "temperature": 0.5
    }


    response = requests.post(api_url, headers = request_headers, json = request_data)

    # check if the request was successful
    if response.status_code == 200:
        print(response.json())
        print(response.json()["choices"][0]["text"])
        response_text = response.json()["choices"][0]["text"]


        st.subheader('The Response is: ')
        st.write(response_text)

        # write this to a seperate python file
        with open("pythonFile.py", "w") as file:
            file.write(response_text)
    else:
        print(f"request Failed with status code: {str(response.status_code)}")

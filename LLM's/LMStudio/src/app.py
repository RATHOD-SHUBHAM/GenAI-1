import os
import openai

openai.api_base = "http://localhost:1234/v1"
openai.api_key = ""


# completion = openai.ChatCompletion.create(
#   model="local-model", # this field is currently unused
#   messages=[
#     {"role": "user", "content": "Can you give me python code to import csv file."}
#   ]
# )
#
# print(completion.choices[0].message)

model = 'LLama2'

def get_completion(prompt, model , temperature = 0.0):
    completion = openai.ChatCompletion.create(
      model=model, # this field is currently unused
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    return completion.choices[0].message['content']


if __name__ == '__main__':
    prompt = "What is AI."
    response = get_completion(prompt=prompt, model=model,temperature=0)
    print(response)

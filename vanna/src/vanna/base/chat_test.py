import os
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = 'https://embedding-openai-alpha.openai.azure.com/'
AZURE_OPENAI_API_KEY = 'f2a7a10f1807436c9412ff477469decf'

client = AzureOpenAI(
  api_key=AZURE_OPENAI_API_KEY,
  api_version="2024-02-01",
  azure_endpoint=AZURE_OPENAI_ENDPOINT
)

deployment_name = 'gpt-35-turbo'  # This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.

# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for ice cream shop '
response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": start_phrase}
    ]
)
print(response.choices[0].message.content)

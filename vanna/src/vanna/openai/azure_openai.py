import os
from openai import AzureOpenAI
from .base import VannaBase

AZURE_OPENAI_ENDPOINT = 'https://embedding-openai-alpha.openai.azure.com/'
AZURE_OPENAI_API_KEY = 'f2a7a10f1807436c9412ff477469decf'


class OpenAI_Chat(VannaBase):
  def __init__(self, client=None, config=None):
    VannaBase.__init__(self, config=config)

    # default parameters - can be overrided using config
    self.temperature = 0.7
    self.max_tokens = 500

    if client is not None:
      self.client = client
      return

    if config is None and client is None:
      self.client = AzureOpenAI(
                      api_key=AZURE_OPENAI_API_KEY,
                      api_version="2024-02-01",
                      azure_endpoint=AZURE_OPENAI_ENDPOINT
                    )
      return

    if "api_key" in config:
      self.client = AzureOpenAI(api_key=config["api_key"])

  def system_message(self, message: str) -> any:
    return {"role": "system", "content": message}

  def user_message(self, message: str) -> any:
    return {"role": "user", "content": message}

  def assistant_message(self, message: str) -> any:
    return {"role": "assistant", "content": message}

  def submit_prompt(self, prompt, **kwargs) -> str:
    if prompt is None:
      raise Exception("Prompt is None")

    if len(prompt) == 0:
      raise Exception("Prompt is empty")

    # Count the number of tokens in the message log
    # Use 4 as an approximation for the number of characters per token
    num_tokens = 0
    for message in prompt:
      num_tokens += len(message["content"]) / 4


    if num_tokens > 3500:
      model = "gpt-3.5-turbo-16k"
    else:
      model = "gpt-3.5-turbo"

    print(f"Using model {model} for {num_tokens} tokens (approx)")
    response = self.client.chat.completions.create(
      model=model,
      messages=prompt,
      max_tokens=self.max_tokens,
      stop=None,
      temperature=self.temperature,
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
      if "text" in choice:
        return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content

if __name__ == "__main__":
  aoi = OpenAI_Chat()
  aoi.system_message('hello you are a data anlayst')

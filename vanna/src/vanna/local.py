from .chromadb.azure_ai_search import AzureAISearch
from .openai.azure_openai import AzureOpenAI


class LocalContext_OpenAI(AzureAISearch, AzureOpenAI):
    def __init__(self, config=None):
        AzureAISearch.__init__(self, config=config)
        AzureOpenAI.__init__(self, config=config)


if __name__ == "__main__":
  vanna = LocalContext_OpenAI()
  print(vanna.system_message('hello'))

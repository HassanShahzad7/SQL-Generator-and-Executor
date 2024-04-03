from azure_ai_search import AzureAISearch


class LocalContext_OpenAI(AzureAISearch):
    def __init__(self, config=None):
        AzureAISearch.__init__(self, config=config)


if __name__ == "__main__":
    vanna = LocalContext_OpenAI()
    # print(vanna.system_message('hello'))
    query = vanna.generate_sql("What is the salary of a Data Scientist, considering the table to get the data from is 'employees'")
    print(query)
    conn = vanna.getconn()
    print(vanna.execute_query(query, conn))
    # print(type(query))
    # print(vanna.execute_query(query))

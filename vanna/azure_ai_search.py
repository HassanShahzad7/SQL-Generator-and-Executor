from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from google.cloud.sql.connector import Connector
import sqlalchemy
import pandas as pd
import uuid, requests
import json
from typing import List
from openai import AzureOpenAI
# from ..base import VannaBase
from dotenv import load_dotenv
import os

AZURE_OPENAI_ENDPOINT = 'https://embedding-openai-alpha.openai.azure.com/'
AZURE_OPENAI_API_KEY = 'f2a7a10f1807436c9412ff477469decf'
SERVICE_NAME = 'vanna-search'
# ADMIN_KEY='EM3AgQnBBsLon6kOOYxnVEpLhzHrHhj3X2feGDtkxTAzSeBkpMcv'
INDEX_NAME = 'vanna-index'
API_VERSION = '2020-06-30'
url = f"https://{SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}?api-version={API_VERSION}"
INSTANCE_CONNECTION_NAME = 'testanalyticsplatform:europe-west3:data-monkey-test-2'
DB_USER = 'postgres'
DB_PASS = 'Datamonkeytest123'
DB_NAME = 'postgres'

search_endpoint = f"https://{SERVICE_NAME}.search.windows.net/"

service_name = 'vanna-search'
ADMIN_KEY = 'EM3AgQnBBsLon6kOOYxnVEpLhzHrHhj3X2feGDtkxTAzSeBkpMcv'
api_version = "2020-06-30"


class AzureAISearch():

    def generate_embeddings(self, text, model="text-embedding-ada-002"): # model = "deployment_name"
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    def getconn(self):
        connector = Connector()
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME
        )
        return conn

    def loading_tables(self, connection, table_name):
        # create connection pool with 'creator' argument to our connection object function
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connection,
        )
        df = pd.read_sql_query(f"SELECT * from {table_name}", engine)
        return df

    def preprocessing_tables(self, df):
        self.df = self.loading_tables(self.getconn)
        df['empid'] = df['empid'].astype('str')
        df['title_embedded'] = df["title"].apply(lambda x: self.generate_embeddings (x, model='text-embedding-ada-002'))
        return df

    def ai_search_index(self, df):
        # Define the headers, including the admin API key
        headers = {
            'Content-Type': 'application/json',
            'api-key': ADMIN_KEY
        }
        payload = {
            "name": INDEX_NAME,
            "fields": [
                {"name": "empid", "type": "Edm.String", "key": True, "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "name", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "title", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "salary", "type": "Edm.Double", "searchable": False, "filterable": True, "sortable": True,
                 "facetable": True},
                {"name": "joining_date", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "title_embedded", "type": "Collection(Edm.Double)"}
            ]
        }
        # Send the request to create the index
        response = requests.put(url, headers=headers, json=payload)

        # Check the response
        if response.status_code in [201, 204]:
            print(f"Index {INDEX_NAME} created successfully.")
        else:
            print(f"Error creating index: {response.json()}")

        # documents = df.to_dict(orient='records')
        # Initialize the SearchClient
        search_client = SearchClient(endpoint=f"https://{SERVICE_NAME}.search.windows.net/",
                                     index_name=INDEX_NAME,
                                     credential=AzureKeyCredential(ADMIN_KEY))

        # Upload documents to the index
        # try:
        #     result = search_client.upload_documents(documents=documents)
        #     print("Batch of documents uploaded successfully.")
        # except Exception as e:
        #     print(f"Error uploading documents: {e}")

        # Perform a search to get all documents. Adjust the search_text and select parameters as needed
        results = search_client.search(search_text="data",
                                       select="*")  # '*' for search_text retrieves all documents, '*' in select retrieves all fields

        # Initialize a counter
        count = 0

        # Iterate over the search results and print them. Stop after 10 results.
        for result in results:
            print(result)
            count += 1
            if count == 10:
                break

    def add_question_sql(self, question: str, sql: str, **kwargs):
        index_name = 'index-sql'
        endpoint = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version={api_version}"
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        headers = {
            'Content-Type': 'application/json',
            'api-key': ADMIN_KEY
        }
        payload = {
            "name": index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "question", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "sql", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True}
            ]
        }
        # Send the request to create the index
        response = requests.put(endpoint, headers=headers, json=payload)

        # Check the response
        if response.status_code in [201, 204]:
            print(f"Index {index_name} created successfully.")
        else:
            print(f"Error creating index: {response.json()}")

        # Create a JSON object containing the question and SQL
        question_sql_json = {
            "id": str(uuid.uuid4()) + "-sql",
            "question": question,
            "sql": sql
        }

        search_client_sql.upload_documents(documents=[question_sql_json])
        print('Documents uploaded')
        return search_client_sql

    def add_ddl(self, ddl: str, **kwargs):
        index_name = 'index-ddl'
        endpoint = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version={api_version}"
        search_client_endpoint = 'https://{service_name}.search.windows.net/'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))
        headers = {
            'Content-Type': 'application/json',
            'api-key': ADMIN_KEY
        }
        payload = {
            "name": index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "ddl", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True}
            ]
        }
        # Send the request to create the index
        response = requests.put(endpoint, headers=headers, json=payload)

        # Check the response
        if response.status_code in [201, 204]:
            print(f"Index {index_name} created successfully.")
        else:
            print(f"Error creating index: {response.json()}")
        # Upload the DDL statement to Azure AI Search
        document = {
            "id": str(uuid.uuid4()) + "-ddl",
            "ddl": ddl,
            **kwargs  # Include any additional fields
        }
        search_client_ddl.upload_documents(documents=[document])
        print('document uploaded')
        return search_client_ddl

    def add_documentation(self, documentation: str, **kwargs):
        index_name = 'index-doc'
        endpoint = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version={api_version}"
        search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))
        headers = {
            'Content-Type': 'application/json',
            'api-key': ADMIN_KEY
        }
        payload = {
            "name": index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "documentation", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True}
            ]
        }
        # Send the request to create the index
        response = requests.put(endpoint, headers=headers, json=payload)

        # Check the response
        if response.status_code in [201, 204]:
            print(f"Index {index_name} created successfully.")
        else:
            print(f"Error creating index: {response.json()}")

        # Upload the documentation to Azure AI Search
        document = {
            "id": str(uuid.uuid4()) + "-doc",
            "documentation": documentation,
            **kwargs  # Include any additional fields
        }
        search_client_doc.upload_documents(documents=[document])
        print('document uploaded')
        return search_client_doc

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        index_name = 'index-sql'
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_ddl = 'index-ddl'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_ddl,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_doc = 'index-doc'
        search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_doc,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        df = pd.DataFrame()
        results_sql = search_client_sql.search(search_text="*", select="*")
        results_ddl = search_client_ddl.search(search_text="*", select="*")
        results_doc = search_client_doc.search(search_text="*", select="*")

        for document in results_sql:
            # print(result)
            # document = result.as_dict()
            df_sql = pd.DataFrame({
                "id": [document["id"]],
                "question": [document["question"]],
                "content": [document["sql"]]
            })
            df = pd.concat([df, df_sql], ignore_index=True)

        for document in results_ddl:
            df_ddl = pd.DataFrame({
                "id": [document["id"]],
                "question": [None],
                "content": [document["ddl"]]
            })
            df = pd.concat([df, df_ddl], ignore_index=True)

        for document in results_doc:
            df_doc = pd.DataFrame({
                "id": [document["id"]],
                "question": [None],
                "content": [document["documentation"]]
            })
            df = pd.concat([df, df_doc], ignore_index=True)

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if not id:
            print("Invalid document ID")
            return False

        index_name = 'index-sql'
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_ddl = 'index-ddl'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_ddl,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_doc = 'index-doc'
        search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_doc,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        if id.endswith("-sql"):
            if search_client_sql.get_document(id) is not None:
                search_client_sql.delete_documents(documents=[{"id": id}])
                print(f'Document deleted {id}')
                return True
            else:
                print(f'Document with ID {id} does not exist')
                return False

        elif id.endswith("-ddl"):
            if search_client_ddl.get_document(id) is not None:
                search_client_ddl.delete_documents(documents=[{"id": id}])
                print(f'Document deleted {id}')
                return True
            else:
                print(f'Document with ID {id} does not exist')
                return False

        elif id.endswith("-doc"):
            if search_client_doc.get_document(id) is not None:
                search_client_doc.delete_documents(documents=[{"id": id}])
                print(f'Document deleted {id}')
                return True
            else:
                print(f'Document with ID {id} does not exist')
                return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (dict): The dictionary containing the query results.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        if query_results is None or "value" not in query_results:
            return []

        documents = query_results["value"]

        if len(documents) == 1 and isinstance(documents[0], list):
            try:
                documents = [json.loads(doc) for doc in documents[0]]
            except Exception as e:
                return documents[0]

        return documents

    def get_similar_question_sql(self, question: str, top_n, **kwargs) -> List[dict]:
        """
        Get similar questions based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of similar questions.
        """

        index_name = 'index-sql'
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        results = search_client_sql.search(search_text=question,
                                           select="*",
                                           top=top_n)

        return results

    def get_related_ddl(self, question: str, top_n, **kwargs) -> List[dict]:
        """
        Get related DDL based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of related DDL documents.
        """

        index_ddl = 'index-ddl'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_ddl,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        results = search_client_ddl.search(search_text=question,
                                           select="*",
                                           top=top_n)

        return results

    def get_related_documentation(self, question: str, top_n, **kwargs) -> List[dict]:
        """
        Get related documentation based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of related documentation documents.
        """

        index_doc = 'index-doc'
        search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_doc,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        results = search_client_doc.search(search_text=question,
                                           select="*",
                                           top=top_n)

        return results

if __name__ == "__main__":
    # print(AZURE_OPENAI_ENDPOINT)
    azure_ai_search = AzureAISearch()
    # conn = azure_ai_search.getconn()
    # data = azure_ai_search.loading_tables(conn, 'employees')
    # print(data)
    # azure_ai_search.add_question_sql('What is the salary of John Doe?', "select salary from employees where name = 'John Doe'")
    # azure_ai_search.add_ddl('CREATE TABLE employees (name VARCHAR(255),title VARCHAR(255),salary NUMERIC);')
    # azure_ai_search.add_documentation('This table contains the information about the employees in a company, it has multiple columns namely, emp_id, title of the employee, and his salary')
    # print(azure_ai_search.get_training_data())
    # azure_ai_search.remove_training_data('7122598d-95b8-4ac6-b08f-be6484cbbfe7-doc')
    # similar_results = azure_ai_search.get_similar_question_sql('What is the salary for Alice Johnson', 2)
    # similar_results = azure_ai_search.get_related_ddl('create table', 2)
    # similar_results = azure_ai_search.get_related_documentation('This table contain employee data', 2)
    # for result in similar_results:
    #     print(result)
    # df = azure_ai_search.preprocessing_tables(table)
    # azure_ai_search.ai_search_index(df, ADMIN_KEY)


    # Example usage
    # question = "How to retrieve data from a table?"
    # sql = "SELECT * FROM employee;"
    # additional_fields = {"category": "SQL Query"}

    # Add the question and SQL to Azure AI Search
    # document_id = azure_ai_search.add_question_sql(question, sql, **additional_fields)
    # print("Document ID:", document_id)
# Importing Libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from google.cloud.sql.connector import Connector
import sqlalchemy
import pandas as pd
import uuid, requests
import json
from typing import List
from openai import AzureOpenAI
from base import VannaBase
import time

SERVICE_NAME = 'xxx'
INDEX_NAME = 'xxx'
API_VERSION = 'xxx'
url = f"https://{SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}?api-version={API_VERSION}"

search_endpoint = f"https://{SERVICE_NAME}.search.windows.net/"

service_name = 'vanna-search'
ADMIN_KEY = 'xxx'
api_version = "xxxx"

AZURE_OPENAI_ENDPOINT = 'xxx'
AZURE_OPENAI_API_KEY = 'xxx'
class AzureAISearch(VannaBase):

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
                api_version="2023-05-15",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            return

        if "api_key" in config:
            self.client = AzureOpenAI(api_key=config["api_key"])

    # Method to Generate Textual Embeddings
    def generate_embedding(self, text, model="text-embedding-ada-002"): # model = "deployment_name"
        """
        This method generates the embedding of a given text using the specified Azure OpenAI model.

        Args:
            text (str): The text string to generate the embedding for.
            model (str): The Azure OpenAI model to use for embedding generation. Default is "text-embedding-ada-002".

        Returns:
            list: The embedding vector of the input text as a list of floats.

        """

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    # Method to load all the tables in the database
    def loading_tables(self, connection, table_name):
        """
        This method loads a table from a PostgreSQL database into a pandas DataFrame.

        Args:
            connection: A database connection object used to connect to PostgreSQL.
            table_name (str): The name of the database table to load.

        Returns:
            DataFrame: A pandas DataFrame containing all rows from the specified table.
        """

        # create connection pool with 'creator' argument to our connection object function
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connection,
        )
        df = pd.read_sql_query(f"SELECT * from {table_name}", engine)
        return df

    # This method can be used if textual embedding is required
    def preprocessing_tables(self, df):
        """
        This method preprocesses a pandas DataFrame by loading data from a table and then performing type casting and embedding generation on specific columns.

        Args:
            df (DataFrame): The initial pandas DataFrame that is provided for preprocessing.

        Returns:
            DataFrame: The preprocessed pandas DataFrame with string-converted 'empid' and generated embeddings for the 'title' column.
        """

        self.df = self.loading_tables(self.getconn)
        df['empid'] = df['empid'].astype('str')
        df['title_embedded'] = df["title"].apply(lambda x: self.generate_embedding (x, model='text-embedding-ada-002'))
        return df

    # Method to create an index on Azure AI Search
    def ai_search_index(self, df):
        """
        This method sets up and manages an AI search index using Azure's Search services, performs document uploads, and executes a search query.

        Args:
            df (DataFrame): The DataFrame from which data is used to populate the search index.

        Returns:
            None: This method primarily performs API calls to Azure and handles search operations without returning a value. Outputs results directly via print statements.
        """

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

    # Method to Add Question SQL pair to the Azure AI Search
    def add_question_sql(self, db_id: str, db_name: str, org_id: str, question: str, sql: str, **kwargs):
        """
        This method adds questions along with their corresponding SQL statements to an Azure Search index. It initializes the search client, creates or updates the index, and uploads the documents.

        Args:
            db_id (str): Unique identifier for the database.
            db_name (str): Name of the database.
            org_id (str): Unique identifier for the organization.
            question (str): The question to index.
            sql (str): The SQL statement corresponding to the question.
            **kwargs: Additional keyword arguments that can be used for extending functionality.

        Returns:
            SearchClient: The search client used for the operation, which may be used for further operations like querying.
        """

        index_name = 'index_sql'
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
                {"name": "database_id", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "database_name", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "organization_id", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "question", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "sql", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True}
            ]
        }
        # Send the request to create the index
        # response = requests.put(endpoint, headers=headers, json=payload)
        #
        # # Check the response
        # if response.status_code in [201, 204]:
        #     print(f"Index {index_name} created successfully.")
        # else:
        #     print(f"Error creating index: {response.json()}")

        # Create a JSON object containing the question and SQL
        question_sql_json = {
            "id": str(uuid.uuid4()) + "-sql",
            "database_id": db_id,
            "database_name": db_name,
            "organization_id": org_id,
            "question": question,
            "sql": sql
        }

        search_client_sql.upload_documents(documents=[question_sql_json])
        print('Documents uploaded')
        return search_client_sql

    # Method to Add DDL to the Azure AI Search
    def add_ddl(self, db_id: str, db_name: str, org_id: str, ddl: str, **kwargs):
        """
        This method adds DDL statements to an Azure Search index. It sets up the search client, potentially creates or updates the index, and uploads the DDL document.

        Args:
            db_id (str): Unique identifier for the database.
            db_name (str): Name of the database.
            org_id (str): Unique identifier for the organization.
            ddl (str): The DDL (Data Definition Language) statement to index.
            **kwargs: Additional keyword arguments for extending document properties.

        Returns:
            SearchClient: The search client used for the operation, which may be used for further operations like querying.
        """

        index_name = 'index_ddl'
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
                {"name": "database_id", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "database_name", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "organization_id", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "ddl", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True}
            ]
        }
        # Send the request to create the index
        # response = requests.put(endpoint, headers=headers, json=payload)
        #
        # # Check the response
        # if response.status_code in [201, 204]:
        #     print(f"Index {index_name} created successfully.")
        # else:
        #     print(f"Error creating index: {response.json()}")
        # Upload the DDL statement to Azure AI Search
        document = {
            "id": str(uuid.uuid4()) + "-ddl",
            "database_id": db_id,
            "database_name": db_name,
            "organization_id": org_id,
            "ddl": ddl,
            **kwargs  # Include any additional fields
        }
        search_client_ddl.upload_documents(documents=[document])
        print('document uploaded')
        return search_client_ddl

    # Method to Add Documentation to the Azure AI Search
    def add_documentation(self, db_id: str, db_name: str, org_id: str, documentation: str, **kwargs):
        """
        This method adds database documentation to an Azure Search index. It establishes a search client, potentially creates or updates the index, and uploads the documentation document.

        Args:
            db_id (str): Unique identifier for the database.
            db_name (str): Name of the database.
            org_id (str): Unique identifier for the organization.
            documentation (str): The text of the documentation to be indexed.
            **kwargs: Additional keyword arguments for extending document properties.

        Returns:
            SearchClient: The search client used for the operation, which may be used for further operations like querying.
        """

        index_name = 'index_doc'
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
                {"name": "database_id", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True},
                {"name": "database_name", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "organization_id", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True},
                {"name": "documentation", "type": "Edm.String", "searchable": True, "filterable": True,
                 "sortable": True}
            ]
        }
        # Send the request to create the index
        # response = requests.put(endpoint, headers=headers, json=payload)

        # Check the response
        # if response.status_code in [201, 204]:
        #     print(f"Index {index_name} created successfully.")
        # else:
        #     print(f"Error creating index: {response.json()}")

        # Upload the documentation to Azure AI Search
        document = {
            "id": str(uuid.uuid4()) + "-doc",
            "database_id": db_id,
            "database_name": db_name,
            "organization_id": org_id,
            "documentation": documentation,
            **kwargs  # Include any additional fields
        }
        search_client_doc.upload_documents(documents=[document])
        print('document uploaded')
        return search_client_doc

    # Method to return all the training data stored in Azure AI Search, it returns question-sql, ddl and documentation data
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        This method retrieves and compiles training data from multiple Azure Search indexes into a single pandas DataFrame. It accesses three indexes—SQL, DDL, and documentation—and aggregates the results.

        Args:
            **kwargs: Additional keyword arguments that can be used to customize the search queries.

        Returns:
            DataFrame: A pandas DataFrame containing combined data from SQL, DDL, and documentation indexes, structured with columns for ID, question (if applicable), and content.
        """

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

    # Method to delete a single training data in Azure Ai Search by specifying an ID
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        This method removes a document from one of three Azure Search indexes based on the document ID suffix indicating the type of document (SQL, DDL, or documentation).

        Args:
            id (str): The unique identifier of the document to be removed. The suffix ("-sql", "-ddl", "-doc") indicates the index from which the document will be deleted.
            **kwargs: Additional keyword arguments for future customization of the function.

        Returns:
            bool: True if the document is successfully deleted, False if the document ID is invalid or the document does not exist.
        """

        if not id:
            print("Invalid document ID")
            return False

        index_name = 'index_sql'
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_ddl = 'index_ddl'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_ddl,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        index_doc = 'index_doc'
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
                print('check 4')
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

    # Method to Fetch similar question-sql data stored to the user query provided
    def get_similar_question_sql(self, question: str, org_id: str, **kwargs) -> List[dict]:
        """
        Get similar questions based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            org_id(str): The unique id for each organization to filter out organization's data
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of similar questions.
        """

        index_name = 'index_sql'
        search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_name,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        # Use the $filter parameter to search only within the specified organization_id
        org_id = int(org_id)
        filter_query = f"organization_id eq '{org_id}'"
        print(filter_query)
        search_results = search_client_sql.search(search_text=question,
                                            select="*",
                                            filter=filter_query,
                                            query_type='semantic',  # Enables full query capabilities
                                            semantic_configuration_name = 'semantic_sql',
                                            # search_mode='all', # Hybrid search: uses both keyword and semantic search
                                            top=3)

        # Initialize an empty list to hold the results
        results_list = []

        # Iterate over the search results and add each result to the list
        for result in search_results:
            # Convert the result to a dictionary and append it to the results_list
            results_list.append(dict(result))
        print(results_list)
        # Return the populated list of results
        return results_list

    # Method to fetch related ddl data stored to the user query provided
    def get_related_ddl(self, question: str, org_id: str, **kwargs) -> List[dict]:
        """
        Get related DDL based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            org_id(str): The unique id for each organization to filter out organization's data
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of related DDL documents.
        """

        index_ddl = 'index_ddl'
        search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                        index_name=index_ddl,
                                        credential=AzureKeyCredential(ADMIN_KEY))

        # Use the $filter parameter to search only within the specified organization_id
        filter_query = f"organization_id eq '{org_id}'"

        search_results = search_client_ddl.search(search_text=question,
                                            select="*",
                                            filter=filter_query,
                                            query_type='simple',  # Enables full query capabilities
                                            semantic_configuration_name='semantic_ddl',
                                            # search_mode='all',
                                            top=3)

        # Initialize an empty list to hold the results
        results_list = []

        # Iterate over the search results and add each result to the list
        for result in search_results:
            # Convert the result to a dictionary and append it to the results_list
            results_list.append(dict(result))
        print(results_list)
        # Return the populated list of results
        return results_list

    # Method to fetch related documentation data stored to the user query provided
    def get_related_documentation(self, question: str, org_id: str, **kwargs) -> List[dict]:
        """
        Get related documentation based on the provided question using Azure AI Search.

        Args:
            question (str): The query question.
            org_id(str): The unique id for each organization to filter out organization's data
            **kwargs: Additional keyword arguments.

        Returns:
            List[dict]: List of related documentation documents.
        """

        index_doc = 'index_doc'
        search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                         index_name=index_doc,
                                         credential=AzureKeyCredential(ADMIN_KEY))

        # Use the $filter parameter to search only within the specified organization_id
        filter_query = f"organization_id eq '{org_id}'"

        search_results = search_client_doc.search(search_text=question,
                                            select="*",
                                            filter=filter_query,
                                            query_type='simple',  # Enables full query capabilities
                                            semantic_configuration_name='semantic_doc',
                                            # search_mode='all',
                                            top=3)

        # Initialize an empty list to hold the results
        results_list = []

        # Iterate over the search results and add each result to the list
        for result in search_results:
            # Convert the result to a dictionary and append it to the results_list
            results_list.append(dict(result))
        print(results_list)
        # Return the populated list of results
        return results_list

    # Method to pass system message to the LLM
    def system_message(self, message: str) -> any:
        """
        This method creates a system message in a standardized format.

        Args:
            message (str): The message content to be formatted.

        Returns:
            dict: A dictionary representing the system message, including role and content.
        """

        return {"role": "system", "content": message}

    # Method to pass user message to the LLM
    def user_message(self, message: str) -> any:
        """
        This method creates a user message in a standardized format.

        Args:
            message (str): The message content to be formatted.

        Returns:
            dict: A dictionary representing the user message, including role and content.
        """

        return {"role": "user", "content": message}

    # Method to pass user message to the LLM
    def assistant_message(self, message: str) -> any:
        """
        This method creates an assistant message in a standardized format.

        Args:
            message (str): The message content to be formatted.

        Returns:
            dict: A dictionary representing the assistant message, including role and content.
        """

        return {"role": "assistant", "content": message}

    # Submits the prompt enter by the user to the LLM
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        This method submits a chat prompt to a chat model and retrieves the response.

        Args:
            prompt (list of dicts): A list of message dictionaries, each representing a part of the conversation context.
            **kwargs: Additional keyword arguments to pass along with the chat completion request, such as specific settings for the model.

        Returns:
            str: The text content of the first response from the chatbot that contains text. If no such response is found, returns the content of the first response, which may be empty.

        Raises:
            Exception: If the 'prompt' is None or empty.
        """

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
            model = "gpt-35-turbo"
        else:
            model = "gpt-35-turbo"

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
                # print(choice.text)
                return choice.text
        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

    # Creates a connection with the Database
    def getconn(self, org_id: str, db_name: str):
        """
        This method establishes a database connection using credentials and configurations stored in a JSON file for a specified organization and database name.

        Args:
            org_id (str): The unique identifier of the organization for which to establish a connection.
            db_name (str): The name of the database to connect to.

        Returns:
            Connector: A database connection object configured according to the organization's specific database credentials and settings.

        Raises:
            ValueError: If an unsupported database type is specified in the JSON configuration.
        """

        start_time = time.time()

        # Load the credentials from JSON file
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)

        # Access the specific organization's credentials
        org_credentials_list = credentials['organization_id'][org_id]
        # Determine the driver based on the DATABASE type
        # Find the matching database configuration by database_names
        for org_credentials in org_credentials_list:
            print('connector object start')
            print(org_credentials)
            if org_credentials['database_names'] == db_name:
                # Determine the driver based on the DATABASE type
                if org_credentials['DATABASE'].lower() == 'postgres':
                    db_type = 'pg8000'
                elif org_credentials['DATABASE'].lower() == 'mysql':
                    db_type = 'pymysql'
                else:
                    raise ValueError("Unsupported database type specified in the JSON configuration.")
                connector = Connector()
                print(org_credentials['INSTANCE_CONNECTION_NAME'])
                print(db_type)
                print(org_credentials['DB_USER'])
                print(org_credentials['DB_PASS'])
                print(org_credentials['DB_NAME'])
                conn = connector.connect(
                    org_credentials['INSTANCE_CONNECTION_NAME'],
                    db_type,
                    user=org_credentials['DB_USER'],
                    password=org_credentials['DB_PASS'],
                    db=org_credentials['DB_NAME']
                )
                print('connector object done')
                end_time = time.time()  # Record the end time
                execution_time = end_time - start_time  # Calculate total execution time

                print(f"Establishing Connection took: {execution_time} seconds")
                return conn

    # Executes the Query provided by the LLM
    def execute_query(self, query, conn, org_id, db_name):
        """
        This method executes a SQL query against a specified database using credentials and configurations from a JSON file. It supports multiple database types and uses SQLAlchemy for connection management.

        Args:
            query (str): The SQL query to be executed.
            conn: The database connection object.
            org_id (str): The unique identifier of the organization whose database is being queried.
            db_name (str): The name of the database against which the query is executed.

        Returns:
            DataFrame: A pandas DataFrame containing the results of the executed SQL query.

        Raises:
            ValueError: If an unsupported database type is specified in the JSON configuration.
        """

        start_time = time.time()
        # Load the credentials from JSON file
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)

            # Access the specific organization's credentials list
            org_credentials_list = credentials['organization_id'][org_id]

            # Find the matching database configuration by database_names
            for org_credentials in org_credentials_list:
                if org_credentials['database_names'] == db_name:
                    # Determine the driver and dialect based on the DATABASE type
                    if org_credentials['DATABASE'].lower() == 'postgres':
                        # db_type = 'pg8000'
                        # db_dialect = 'postgres'
                        engine = sqlalchemy.create_engine(
                            # Build the SQLAlchemy connection string
                            "postgresql+pg8000://",
                            creator=lambda: conn,  # Pass the connection object directly
                        )
                        return pd.read_sql_query(query, engine)
                    elif org_credentials['DATABASE'].lower() == 'mysql':
                        db_type = 'pymysql'
                        db_dialect = 'mysql'
                    else:
                        raise ValueError("Unsupported database type specified in the JSON configuration.")

                    connection_string = f"{db_dialect}+{db_type}://"
                    print(connection_string)
                    engine = sqlalchemy.create_engine(
                        # Build the SQLAlchemy connection string
                        connection_string,
                        creator=lambda: conn,  # Pass the connection object directly
                    )
                    end_time = time.time()  # Record the end time
                    execution_time = end_time - start_time  # Calculate total execution time

                    print(f"Executing Query took: {execution_time} seconds")
                    return pd.read_sql_query(query, engine)



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
    azure_ai_search.remove_training_data("527417ee-b37e-4b48-9fd3-3bcd04d8add5-sql")
    # similar_results = azure_ai_search.get_similar_question_sql('What is the salary for Alice Johnson')
    # similar_results = azure_ai_search.get_related_ddl('create table')
    # similar_results = azure_ai_search.get_related_documentation('This table contain employee data')
    # print(similar_results)
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

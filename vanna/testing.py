from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import uuid, requests
import pandas as pd
from typing import List

service_name = 'vanna-search'
ADMIN_KEY = 'EM3AgQnBBsLon6kOOYxnVEpLhzHrHhj3X2feGDtkxTAzSeBkpMcv'
api_version = "2020-06-30"


def add_question_sql(question: str, sql: str):

    index_name = 'index-sql'
    endpoint = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version={api_version}"
    # search_client_endpoint = 'https://{service_name}.search.windows.net/'
    search_client_sql = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                 index_name=index_name,
                                 credential=AzureKeyCredential(ADMIN_KEY))
    # index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(ADMIN_KEY))

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


def add_ddl(ddl: str, **kwargs):

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


def add_documentation(documentation: str, **kwargs):

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
            {"name": "documentation", "type": "Edm.String", "searchable": True, "filterable": True, "sortable": True}
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


def get_training_data():

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

def remove_training_data(id: str, **kwargs) -> bool:
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

def get_similar_question_sql(question: str, top_n, **kwargs) -> List[dict]:
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

def get_similar_question_doc(question: str, top_n, **kwargs) -> List[dict]:
    """
    Get similar questions based on the provided question using Azure AI Search.

    Args:
        question (str): The query question.
        **kwargs: Additional keyword arguments.

    Returns:
        List[dict]: List of similar questions.
    """
    index_doc = 'index-doc'
    search_client_doc = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                     index_name=index_doc,
                                     credential=AzureKeyCredential(ADMIN_KEY))

    results = search_client_doc.search(search_text=question,
                                   select="*",
                                   top=top_n)

    return results

def get_similar_question_ddl(question: str, top_n, **kwargs) -> List[dict]:
    """
    Get similar questions based on the provided question using Azure AI Search.

    Args:
        question (str): The query question.
        **kwargs: Additional keyword arguments.

    Returns:
        List[dict]: List of similar questions.
    """
    index_ddl = 'index-ddl'
    search_client_ddl = SearchClient(endpoint=f"https://{service_name}.search.windows.net/",
                                     index_name=index_ddl,
                                     credential=AzureKeyCredential(ADMIN_KEY))

    results = search_client_ddl.search(search_text=question,
                                   select="*",
                                   top=top_n)

    return results


if __name__ == "__main__":
    # question = "What is the salary of an average data scientist?"
    # sql = "SELECT avg(salary) FROM employee where title = 'Data Scientist';"

    # Add the question and SQL to Azure AI Search
    # search_client_sql = add_question_sql(question, sql)
    # document_id = add_question_sql(question, sql)
    # print("Document ID:", document_id)

    # ddl = "CREATE TABLE employees (name VARCHAR(255),title VARCHAR(255),salary NUMERIC);"
    # Add the question and SQL to Azure AI Search
    # search_client_ddl = add_ddl(ddl)
    # document_id = add_question_sql(question, sql)
    # print("Document ID:", document_id)

    # doc = "This table contains the information about the employees in a company, it has multiple columns namely, emp_id, title of the employee, and his salary"
    # Add the question and SQL to Azure AI Search
    # search_client_doc = add_documentation(doc)
    # document_id = add_question_sql(question, sql)
    # print("Document ID:", document_id)

    # df = get_training_data()
    # remove_training_data('80149191-c806-4502-a7e3-86996f370776-ddl')
    # print(df)

    similar_results = get_similar_question_sql('What is an average salary of a Chemical Engineer?', 3)
    for result in similar_results:
        print(result)
    # print(similar_results)
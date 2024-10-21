# Importing Libraries
from azure_ai_search import AzureAISearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

# Instantiating FastAPI
app = FastAPI()


# Creating a Pydantic Model to validate the data type of the column
class Query(BaseModel):
    db_id: str
    db_name: str
    org_id: str
    query: str


class Generate(BaseModel):
    query: str


class SQL(BaseModel):
    db_id: str
    db_name: str
    org_id: str
    question: str
    sql: str


# Root endpoint if no path is specified
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Endpoint to Generate and Execute SQL
@app.post("/generate-sql/")
async def generate_sql(query: Generate):
    vanna = AzureAISearch()
    # Assuming Organization 1 is being fetched from the flow of data
    org_id = '1'
    # Fetching the SQL Query and the DB Name that should be used for the query from the returned tuple
    sql_query, db_name = vanna.generate_sql(query.query, org_id)
    print(sql_query)

    # Creating a connection to the database
    conn = vanna.getconn(org_id, db_name)
    # Executing the query received from LLM
    results = vanna.execute_query(sql_query, conn, org_id, db_name)
    # Converting the results to dictionary
    results_list = results.to_dict('records')
    print(results)
    # Formatting the returned json
    response = jsonable_encoder({'data': {'Query': sql_query, 'Results': results_list}})
    return response


@app.post("/add-sql/")
async def add_sql(query: SQL):
    vanna = AzureAISearch()
    # Adding Question-SQL pair data to the vector storage
    vanna.add_question_sql(query.db_id, query.db_name, query.org_id, query.question, query.sql)
    return {"message": "SQL has been added successfully"}


@app.post("/add-ddl/")
async def add_ddl(query: Query):
    vanna = AzureAISearch()
    # Adding DDL data to the vector storage
    vanna.add_ddl(query.db_id, query.db_name, query.org_id, query.query)
    return {"message": "DDL has been added successfully"}


@app.post("/add-doc/")
async def add_doc(query: Query):
    vanna = AzureAISearch()
    # Adding Documentation data to the vector storage
    vanna.add_documentation(query.db_id, query.db_name, query.org_id, query.query)
    return {"message": "Documentation has been added successfully"}
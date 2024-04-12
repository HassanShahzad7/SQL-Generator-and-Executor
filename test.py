from azure_ai_search import AzureAISearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

app = FastAPI()

class Query(BaseModel):
    query: str


class SQL(BaseModel):
    question: str
    sql: str


# if __name__ == "__main__":
#     vanna = AzureAISearch()
#     # print(vanna.system_message('hello'))
#     query = vanna.generate_sql("What is the salary of a Data Scientist, considering the table to get the data from is 'employees'")
#     print(query)
#     conn = vanna.getconn()
#     print(vanna.execute_query(query, conn))
#     # print(type(query))
#     # print(vanna.execute_query(query))

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/generate-sql/")
async def generate_sql(query: Query):
    vanna = AzureAISearch()
    sql_query = vanna.generate_sql(query.query)
    print(sql_query)
    # return {"message": "Endpoint hit successfully"}
    conn = vanna.getconn()
    results = vanna.execute_query(sql_query, conn)
    results_list = results.to_dict('records')
    print(results)
    response = jsonable_encoder({'data': {'Query': sql_query, 'Results': results_list}})
    return response

@app.post("/add-sql/")
async def add_sql(query: SQL):
    vanna = AzureAISearch()
    vanna.add_question_sql(query.question, query.sql)
    return {"message": "SQL has been added successfully"}

@app.post("/add-ddl/")
async def add_ddl(query: Query):
    vanna = AzureAISearch()
    vanna.add_ddl(query.query)
    return {"message": "DDL has been added successfully"}

@app.post("/add-doc/")
async def add_doc(query: Query):
    vanna = AzureAISearch()
    vanna.add_documentation(query.query)
    return {"message": "Documentation has been added successfully"}
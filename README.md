# SQL Generator Project

This project focuses on generating and executing SQL for the natural language input provided by the user. 

Following are the versions used while working with the Project:
- Python    3.12.0
- FastAPI   0.110.1

Following commands must be run to install the libraries

Standard library modules do not need installation (json, os, re, sqlite3, traceback, abc, typing, urllib)

#### Install pandas
```pip install pandas```

#### Install Plotly for interactive graphing
```pip install plotly```

#### Install requests for making HTTP requests
```pip install requests```

#### Install Azure SDK components
```pip install azure-core azure-search-documents azure-ai-textanalytics```

#### Install Google Cloud SQL Connector and SQLAlchemy for database connections
```pip install google-cloud-sql-connector sqlalchemy```

#### Install FastAPI for creating web APIs
```pip install fastapi```

#### Install Pydantic for data validation and settings management for Python
```pip install pydantic```

#### Install Uvicorn to serve the FastAPI application, as FastAPI itself doesn't include a web server
```pip install uvicorn```

## To Run the project
```uvicorn test:app --reload```

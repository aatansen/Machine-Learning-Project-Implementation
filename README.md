<div align="center">
<h1>Machine Learning Project Implementation</h1>
</div>

# **Context**
- [**Context**](#context)
  - [**Day 01**](#day-01)
    - [Project 01: US Visa Approval Prediction](#project-01-us-visa-approval-prediction)
    - [Requirements](#requirements)
    - [Project Overview](#project-overview)
    - [Deployment](#deployment)
    - [Problem Statement](#problem-statement)
    - [Features](#features)
    - [Solution Scope](#solution-scope)
    - [Solution Approach](#solution-approach)
    - [Solution Proposed](#solution-proposed)
    - [Project Setup](#project-setup)
      - [Environment Setup](#environment-setup)
      - [Project Structure](#project-structure)
      - [Project Structure Template](#project-structure-template)
      - [Requirements.txt \& setup.py](#requirementstxt--setuppy)
  - [**Day 02**](#day-02)
    - [Agenda](#agenda)
    - [Database Setup](#database-setup)

## **Day 01**

### Project 01: US Visa Approval Prediction

### Requirements

- [Anaconda](https://www.anaconda.com/download)
- [Git](https://git-scm.com/downloads)
- [VsCode](https://code.visualstudio.com/)

[⬆️ Go to Context](#context)

### Project Overview

- Understanding the Problem Statement
- Understanding the solution
- Code understanding & walkthrough
- Understanding the Deployment

[⬆️ Go to Context](#context)

### Deployment

- Docker
- Cloud Services
- Adding self-hosted runner
- workflows

[⬆️ Go to Context](#context)

### Problem Statement

- US visa approval status
- Given certain set of feature such as continent, education, job_experience, training, employment, current age etc.
- We have to predict weather the application for the visa will be approved or not.

[⬆️ Go to Context](#context)

### Features

- Continent: `Asia`, `Africa`, `North America`, `Europe`, `South America`, `Oceania`
- Eduction: `High School`, `Master's Degree`, `Bachelor's`, `Doctorate`
- Job Experience: `Yes`, `No`
- Required training: `Yes`, `No`
- Number of employees: `15000` to `40000`
- Region of employment: `West`, `Northeast`, `South`, `Midwest`, `Island`
- Prevailing wage: `700` to `70000`
- Contract Tenure: `Hour`, `Year`, `Week`, `Month`
- Full-time: `Yes`, `No`
- Age of company: `15` to `180`

[⬆️ Go to Context](#context)

### Solution Scope

- This can be used on real life by US visa applicants so that they can improve their resume and criteria for the approval process.

[⬆️ Go to Context](#context)

### Solution Approach

- Machine Learning : ML Classification Algorithms
- Deep Learning: Custom ANN with sigmoid activation Function

[⬆️ Go to Context](#context)

### Solution Proposed

> We will be using ML (Machine Learning)

- Load the data from DB
- Perform EDA and feature engineering to select the desirable features.
- Fit the ML classification Algorithm and find out which one performs better.
- Select top few and tune hyperparameters.
- Select the best model based on desired metrics

[⬆️ Go to Context](#context)

### Project Setup

- GitHub Repository
- Requirements
- Template
- Database
  - [MongoDB](https://www.mongodb.com/)
  - Alternative options: [PostgreSQL](https://www.postgresql.org/), [MySQL](https://www.mysql.com/)

[⬆️ Go to Context](#context)

#### Environment Setup

- Create conda `venv`

  ```sh
  conda create python -p venv
  ```

- Activate `venv`

  ```sh
  conda activate ./venv
  ```

> [!NOTE]
>
> - I use the `-p` (path) option instead of -`n` (name) when creating environments.
> - The -`n` flag assigns a **name** to the environment, and conda stores it inside the default directory, e.g., `C:\...\envs\venv`.
> - The `-p` flag specifies the **path** where the environment should be created.
> - Since I provided only a relative path name (`venv`) with `-p`, conda created the environment in the current working directory.
> - And for activating I have to gave full path but the shortcut is using relative path `./venv`

[⬆️ Go to Context](#context)

#### Project Structure

  ```sh
  ├── 📁 Root Path
  ├── 📁 US Visa Approval Prediction/
  │   ├── 📁 components/
  │   │   ├── 🐍 __init__.py
  │   │   ├── 🐍 data_ingestion.py
  │   │   ├── 🐍 data_transformation.py
  │   │   ├── 🐍 data_validation.py
  │   │   ├── 🐍 model_evaluation.py
  │   │   ├── 🐍 model_pusher.py
  │   │   └── 🐍 model_trainer.py
  │   ├── 📁 config/
  │   │   ├── ⚙️ model.yaml
  │   │   └── ⚙️ schema.yaml
  │   ├── 📁 configuration/
  │   │   ├── 🐍 __init__.py
  │   │   └── 🐍 config.py
  │   ├── 📁 constants/
  │   │   ├── 🐍 __init__.py
  │   │   └── 🐍 constants.py
  │   ├── 📁 data/
  │   │   ├── 📁 interim/
  │   │   │   └── 📄 .gitkeep
  │   │   ├── 📁 processed/
  │   │   │   └── 📄 .gitkeep
  │   │   └── 📁 raw/
  │   │        └── 📄 .gitkeep
  │   ├── 📁 entity/
  │   │   ├── 🐍 __init__.py
  │   │   ├── 🐍 artifact_entity.py
  │   │   └── 🐍 config_entity.py
  │   ├── 📁 logger/
  │   │   └── 🐍 __init__.py
  │   ├── 📁 notebooks/
  │   │   └── 📓 exploration.ipynb
  │   ├── 📁 pipeline/
  │   │   ├── 🐍 __init__.py
  │   │   ├── 🐍 prediction_pipeline.py
  │   │   └── 🐍 training_pipeline.py
  │   ├── 📁 tests/
  │   │   ├── 🐍 __init__.py
  │   │   ├── 🐍 test_data_ingestion.py
  │   │   ├── 🐍 test_data_transformation.py
  │   │   └── 🐍 test_model_trainer.py
  │   ├── 📁 utils/
  │   │   ├── 🐍 __init__.py
  │   │   └── 🐍 main_utils.py
  │   └── 🐍 __init__.py
  ├── 📄 .dockerignore
  ├── 🔒 .env
  ├── 🚫 .gitignore
  ├── 📄 .python-version
  ├── 🐳 Dockerfile
  ├── 📖 README.md
  ├── 🐍 app.py
  ├── 🐍 demo.py
  ├── 📄 requirements.txt
  ├── 🐍 setup.py
  └── 🐍 template.py
  ```

[⬆️ Go to Context](#context)

#### Project Structure Template

- Run [template.py](template.py)

  ```py
  py template.py
  ```

- It will generate [Project Structure](#project-structure) files/folders automatically

[⬆️ Go to Context](#context)

#### Requirements.txt & setup.py

- Update [requirements.txt](./US%20Visa%20Approval%20Prediction/requirements.txt) by adding the following dependencies

  ```txt
  boto3==1.40.16
  catboost==1.2.8
  dill==0.4.0
  evidently==0.7.12
  fastapi==0.116.1
  from-root==1.3.0
  httptools==0.6.4
  imbalanced-learn==0.14.0
  Jinja2==3.1.6
  mypy-boto3-s3==1.40.0
  neuro-mf==0.0.5
  pyarrow==21.0.0
  pymongo==4.14.1
  python-dotenv==1.1.1
  python-multipart==0.0.20
  seaborn==0.13.2
  watchfiles==1.1.0
  websockets==15.0.1
  xgboost==3.0.4
  ```

- Now to use folder as module we have added `__init__.py` (*[template.py](template.py) added that while creating all files/folders*)
- Now using [setup.py](./US%20Visa%20Approval%20Prediction/setup.py) we configure the project so that any folder with `__init__.py` file is treated as a package

  ```py
  from setuptools import setup, find_packages

  setup(
      name="us_visa_approval_prediction",
      version="0.0.1",
      author="Tansen",
      author_email="aatansen@gmail.com",
      packages=find_packages()
  )
  ```

- Now add this at the end of the [requirements.txt](./US%20Visa%20Approval%20Prediction/requirements.txt)

  ```sh
  -e .
  ```

- Now run this command to install all dependencies

  ```sh
  pip install -r requirements.txt
  ```

[⬆️ Go to Context](#context)

## **Day 02**

### Agenda

- Database setup (MongoDB Atlas)
- Logging Module
- Exception Module
- Utility Module

[⬆️ Go to Context](#context)

### Database Setup

- Create [MongoDB Account](https://account.mongodb.com/account/register)
- Create New project
- Create cluster and database user
- Get driver code of relevant python version (*I used Stable API:  `3.12 or later`*)
- Get the dataset from Kaggle: [EasyVisa_Dataset](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)
- Using pandas to load the data and do some preprocessing & others check in notebook [exploration.ipynb](./US%20Visa%20Approval%20Prediction/notebooks/exploration.ipynb)
- To save data in MongoDB we have to convert the data into dictionary

  ```py
  data=df.to_dict(orient='records')
  ```

- Add Config in [.env](./US%20Visa%20Approval%20Prediction/.env)

- MongoDB settings

  ```py
  # Get values from environment
  DB_NAME = os.getenv("DB_NAME")
  COLLECTION_NAME = os.getenv("COLLECTION_NAME")
  CONNECTION_URL = os.getenv("CONNECTION_URL")
  ```

- Insert the converted dictionary data in MongoDB

  ```py
  import pymongo
  client = pymongo.MongoClient(CONNECTION_URL)
  data_base = client[DB_NAME]
  collection = data_base[COLLECTION_NAME]
  ```

- Now insert data

  ```py
  records = collection.insert_many(data)
  ```

- If data insert failed due to `timeout error` use `certifi`

  ```py
  import certifi
  client = pymongo.MongoClient(CONNECTION_URL,tlsCAFile=certifi.where())
  ```

  > [!NOTE]
  >
  > As I am using latest version and I did not get any error related to `timeout`
  >
  > If still error persists use different stable version of `pymongo` (e.g: `pymongo>=4.7`)

- Ingest data from MongoDB

  ```py
  records = collection.find().limit(5)
  for i,j in enumerate(records):
    print(f"{i}: {j}")
  ```

  > [!NOTE]
  > In the notebook we can see records output is `cursor object`.
  >
  > A `cursor object` is a `forward-only`, `lazy-loading` iterator returned by `MongoDB` queries (like `find()`), which acts as a pointer to the query results on the server and streams documents in batches until exhausted.It Exhausted after full iteration. Can be converted to list with list(cursor). More memory-efficient than lists. Requires new query to reuse.

- Cursor Object to DataFrame

  ```py
  df = pd.DataFrame(list(collection.find()))
  ```

- Dropping `MongoDB`'s `_id` from column

  ```py
  if '_id' in df.columns.to_list():
      df.drop(columns=['_id'], inplace=True)
  df.head()
  ```

[⬆️ Go to Context](#context)

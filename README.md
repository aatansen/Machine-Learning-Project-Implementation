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
  📁 US Visa Approval Prediction/
  ├── 📁 components/
  │   ├── 🐍 __init__.py
  │   ├── 🐍 data_ingestion.py
  │   ├── 🐍 data_transformation.py
  │   ├── 🐍 data_validation.py
  │   ├── 🐍 model_evaluation.py
  │   ├── 🐍 model_pusher.py
  │   └── 🐍 model_trainer.py
  ├── 📁 config/
  │   ├── ⚙️ model.yaml
  │   └── ⚙️ schema.yaml
  ├── 📁 configuration/
  │   ├── 🐍 __init__.py
  │   └── 🐍 config.py
  ├── 📁 constants/
  │   ├── 🐍 __init__.py
  │   └── 🐍 constants.py
  ├── 📁 entity/
  │   ├── 🐍 __init__.py
  │   ├── 🐍 artifact_entity.py
  │   └── 🐍 config_entity.py
  ├── 📁 logger/
  │   ├── 🐍 __init__.py
  │   └── 🐍 logger.py
  ├── 📁 notebooks/
  │   └── 📓 exploration.ipynb
  ├── 📁 pipeline/
  │   ├── 🐍 __init__.py
  │   ├── 🐍 prediction_pipeline.py
  │   └── 🐍 training_pipeline.py
  ├── 📁 tests/
  │   ├── 🐍 __init__.py
  │   ├── 🐍 test_data_ingestion.py
  │   ├── 🐍 test_data_transformation.py
  │   └── 🐍 test_model_trainer.py
  ├── 📁 utils/
  │   ├── 🐍 __init__.py
  │   └── 🐍 main_utils.py
  ├── 📄 .dockerignore
  ├── 🐳 Dockerfile
  ├── 🐍 __init__.py
  ├── 🐍 app.py
  ├── 🐍 demo.py
  ├── 📄 requirements.txt
  └── 🐍 setup.py
  ```

[⬆️ Go to Context](#context)

#### Project Structure Template

- Run [template.py](template.py)

  ```py
  py template.py
  ```

- It will generate [Project Structure](#project-structure) files/folders automatically

[⬆️ Go to Context](#context)

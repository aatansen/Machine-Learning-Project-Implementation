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
      - [MongoDB settings](#mongodb-settings)
      - [Insert Data to MongoDB](#insert-data-to-mongodb)
      - [Ingest Data from MongoDB](#ingest-data-from-mongodb)
    - [Project Utility](#project-utility)
      - [Logging Functionality](#logging-functionality)
      - [Exception handling](#exception-handling)
      - [Utility functions / Commonly used functions](#utility-functions--commonly-used-functions)
  - [**Day 03**](#day-03)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
      - [Types of Analysis in EDA](#types-of-analysis-in-eda)
      - [Types of Variables in EDA](#types-of-variables-in-eda)
      - [Multicollinearity In EDA](#multicollinearity-in-eda)
    - [Feature Engineering](#feature-engineering)
      - [Feature Extraction](#feature-extraction)
      - [Feature Transform](#feature-transform)
      - [Skewness and Outlier](#skewness-and-outlier)
      - [Column Transformation for Categorical Values](#column-transformation-for-categorical-values)
      - [Handling Imbalanced Data](#handling-imbalanced-data)
    - [Model Training](#model-training)
      - [Train-Test Split](#train-test-split)
      - [List of Models](#list-of-models)
      - [Hyperparameter Tuning](#hyperparameter-tuning)
      - [Final Report](#final-report)

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

[⬆️ Go to Context](#context)

#### MongoDB settings

- Add Config in [.env](./US%20Visa%20Approval%20Prediction/.env)

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

[⬆️ Go to Context](#context)

#### Insert Data to MongoDB

- Insert data

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

[⬆️ Go to Context](#context)

#### Ingest Data from MongoDB

- Ingest data

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

### Project Utility

- Logging functionality
- Exception handling
- Utility functions / Commonly used functions (e.g: YAML file read in any places)

[⬆️ Go to Context](#context)

#### Logging Functionality

- Edit [logger `__init__.py`](./us_visa_approval_prediction/logger/__init__.py) file to save logs
- Test it in [demo.py](./demo.py)

[⬆️ Go to Context](#context)

#### Exception handling

- Edit [exception `__init__.py`](./us_visa_approval_prediction/exception/__init__.py) file to show properly formatted error message by custom exception
- Test it in [demo.py](./demo.py)

[⬆️ Go to Context](#context)

#### Utility functions / Commonly used functions

- Edit utils [main_utils.py](./us_visa_approval_prediction/utils/main_utils.py) here we defined all the commonly used functions

[⬆️ Go to Context](#context)

## **Day 03**

### Exploratory Data Analysis (EDA)

- Full EDA can be found in [02_eda_us_visa.ipynb](./us_visa_approval_prediction/notebooks/02_eda_us_visa.ipynb) inside [notebooks](./us_visa_approval_prediction/notebooks/)

[⬆️ Go to Context](#context)

#### Types of Analysis in EDA

- Univariate Analysis
  - “What does this single column look like?”
  - Look at one feature alone → Age distribution, Visa approval %

- Bivariate Analysis
  - “How does one column affect another?”
  - Look at two features together → Age vs Visa Status
- Multivariate Analysis
  - “How do many columns work together?”
  - Look at many features together → Age + Education + Country → Visa Status

[⬆️ Go to Context](#context)

#### Types of Variables in EDA

- Discrete Variables
  - “Countable steps or categories”
  - Whole numbers, few unique values → Visa Status (Approved/Denied), Education (Bachelor/Master/PhD)

- Continuous Variables
  - “Smooth measurements, can take decimals”
  - Measured values, many unique values → Age, Income, Years of Experience

More Variables:

- Categorical Variables
  - Represent categories or labels, not numbers
  - Can be nominal (no order) or ordinal (ordered)
  - Examples: Country, Gender, Education Level

- Binary Variables
  - Special case of categorical variables with only two categories
  - Examples: Yes/No, True/False, Approved/Denied

- Date/Time Variables
  - Represent time-related information, can be transformed into numeric features
  - Examples: Application Date, Visa Issue Date

- Text Variables
  - Unstructured textual data, can be converted into features using NLP techniques
  - Examples: Job Description, Applicant Remarks

[⬆️ Go to Context](#context)

#### Multicollinearity In EDA

- Definition
  - “When two or more independent variables are highly correlated with each other”
  - Means they provide duplicate information instead of unique insight

- Why It Matters
  - Makes it hard to understand which feature actually influences the target
  - Can cause instability in regression models

- Detection Methods (Numeric)
  - Correlation heatmap → Check if correlation > 0.8 or < -0.8 between features
  - Variance Inflation Factor (VIF) → VIF > 10 usually indicates high multicollinearity

- Detection Methods (Categorical)
  - Cramér’s V → Measures strength of association between two categorical variables (0–1 scale)
  - Chi-square (χ²) Test → Low p-value (<0.05) indicates dependence between categorical variables

[⬆️ Go to Context](#context)

### Feature Engineering

- **Definition:**
  Feature Engineering is the process of creating, transforming, or selecting features (variables) from raw data to improve the performance of machine learning models. It helps the model better capture patterns, relationships, and trends in the data.

- **Key Objectives:**
  - Make raw data more suitable for modeling
  - Handle missing values, outliers, and skewed distributions
  - Transform categorical variables into numerical formats
  - Create new features that capture important patterns or relationships

- **Common Techniques:**
  - **Feature Extraction:** Creating new features from existing data (e.g., extracting year from a date, calculating ratios)
  - **Feature Transformation:** Scaling, normalizing, or transforming features (e.g., log transform, Yeo-Johnson transform)
  - **Feature Encoding:** Converting categorical variables to numeric (Label Encoding, Ordinal Encoding, One-Hot Encoding)
  - **Handling Imbalanced Data:** Techniques like SMOTE to balance classes

[⬆️ Go to Context](#context)

#### Feature Extraction

- Creating new features from existing data to capture important patterns
- Examples:
  - Extracting "year" and "month" from a datetime column
  - Counting number of previous visa applications per applicant
  - Generating ratios, differences, or aggregations from numeric features

[⬆️ Go to Context](#context)

#### Feature Transform

- Transforming features to improve distribution, handle outliers, or scale data
- Examples:
  - Log transformation → reduces right skew
  - Yeo-Johnson transformation → handles skewness and outliers for positive and negative values
  - Standardization (z-score) → mean=0, std=1
  - Min-Max scaling → scales features to [0,1] range

[⬆️ Go to Context](#context)

#### Skewness and Outlier

- Skewness refers to a distortion or asymmetry that deviates from a normal distribution (bell curve)
- Positive (right) skew → tail on the right, mean > median, may indicate extreme high values (outliers)
- Negative (left) skew → tail on the left, mean < median, may indicate extreme low values (outliers)
- A normal distribution has skewness = 0
- Skewed columns (e.g., number of employees, company age) may need transformation
- Power transforms (Box-Cox or Yeo-Johnson) can make skewed data more Gaussian-like
- Useful for addressing heteroscedasticity or modeling assumptions where normality is desired
- Skewness can act as a **red flag for potential outliers**, but visual confirmation with boxplots or histograms is recommended

[⬆️ Go to Context](#context)

#### Column Transformation for Categorical Values

- In machine learning, categorical variables need to be converted into numeric format for models to process them effectively.

- **Label Encoding**
  - Assigns a unique integer to each category
  - Suitable when no ordinal relationship exists, but watch out: it may imply a false ordering
  - Example:

    ```py
    Colors = ['Red', 'Green', 'Blue', 'Red', 'Green']
    LabelEncoder → [2, 1, 0, 2, 1]
    ```

  - Pros: Simple, compact representation
  - Cons: Implies ordinal relationship which may not exist

  - **Ordinal Encoding**
    - Preserves an explicit order among categories
    - Suitable for ordinal variables like education level, rating, or size
    - Example:

      ```py
      Education = ['High School', 'Bachelor', 'Master', 'Ph.D.']
      OrdinalEncoder → [[0], [1], [2], [3]]
      ```

    - Pros: Maintains order information
    - Cons: Assumes linear order; may not capture non-linear relationships

  - **One-Hot Encoding**
    - Converts each category into a binary vector (1 for presence, 0 for absence)
    - Avoids introducing ordinal relationships
    - Example:

      ```py
      Fruit = ['Apple', 'Banana', 'Orange', 'Apple', 'Banana']
      pd.get_dummies →
         Fruit_Apple  Fruit_Banana  Fruit_Orange
         1            0             0
         0            1             0
         0            0             1
         1            0             0
         0            1             0
      ```

    - Pros: No false ordinal relationships
    - Cons: Can lead to high-dimensional feature space with many categories

  - **Conclusion**
    - Choose encoding based on data type and model requirements:
      - Label Encoding → simple, may imply order
      - Ordinal Encoding → preserves natural order
      - One-Hot Encoding → avoids order but increases dimensions

[⬆️ Go to Context](#context)

#### Handling Imbalanced Data

- Imbalanced datasets occur when one class has significantly more samples than another, which can bias model training.

- **Upsampling (Over-sampling)**
  - Increase the number of samples in the minority class by **duplicating existing samples** or creating synthetic ones
  - Example: If class A has 1000 samples and class B has 100, duplicate class B samples to reach 1000

- **Downsampling (Under-sampling)**
  - Reduce the number of samples in the majority class to match the minority class
  - Example: If class A has 1000 samples and class B has 100, randomly remove samples from class A to reach 100

- **SMOTE (Synthetic Minority Over-sampling Technique)**
  - Generates synthetic samples for the minority class to balance the dataset
  - Works by creating new instances along the line segments between existing minority samples
  - Helps improve model performance on minority class predictions
  - Example usage:

    ```py
    from imblearn.over_sampling import SMOTE

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    ```

- Other techniques:
  - Undersampling the majority class
  - Combining over- and under-sampling
  - Using class-weighted algorithms

[⬆️ Go to Context](#context)

### Model Training

#### Train-Test Split

- Before training models, the dataset is split into **training set** and **testing set**
  - Training set → used to train the model
  - Testing set → used to evaluate model performance on unseen data
- Common split ratio: 70:30 or 80:20

#### List of Models

- Multiple models are trained to compare their performance on the dataset.
  - **Random Forest Classifier** → ensemble of decision trees, good for handling non-linear relationships
  - **Decision Tree Classifier** → simple interpretable tree-based model
  - **Gradient Boosting Classifier** → sequential ensemble boosting method
  - **Logistic Regression** → linear model for binary classification
  - **K-Nearest Neighbors (KNN) Classifier** → distance-based model, predicts based on neighbors
  - **XGBoost Classifier** → efficient gradient boosting implementation
  - **CatBoost Classifier** → gradient boosting with categorical feature support
  - **Support Vector Classifier (SVC)** → finds optimal hyperplane to separate classes
  - **AdaBoost Classifier** → ensemble boosting using weighted trees

- **Initial Model Results:**

  | Model Name                | Accuracy |
  | ------------------------- | -------- |
  | Random Forest             | 0.9528   |
  | K-Neighbors Classifier    | 0.9416   |
  | XGBClassifier             | 0.9352   |
  | Decision Tree             | 0.9269   |
  | CatBoosting Classifier    | 0.9261   |
  | Gradient Boosting         | 0.8791   |
  | Support Vector Classifier | 0.8571   |
  | AdaBoost Classifier       | 0.8492   |
  | Logistic Regression       | 0.7256   |

#### Hyperparameter Tuning

- Process of fine-tuning model parameters to improve performance
- Focus on the top-performing models: K-Nearest Neighbors, Random Forest, and XGBClassifier
- Retrain these models using their best-found parameters to maximize accuracy and generalization

- **Tuned Results:**

  | Model Name               | Accuracy |
  | ------------------------ | -------- |
  | KNeighborsClassifier     | 0.9683   |
  | Random Forest Classifier | 0.9528   |
  | XGBClassifier            | 0.9457   |

#### Final Report

- **Best Model:** K-Nearest Neighbors (KNN)
- **Accuracy:** 96.83%

[⬆️ Go to Context](#context)

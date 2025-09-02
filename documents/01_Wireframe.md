<div align="center">
<h1>📑 Wireframe 📑</h1>
</div>

# **Context**
- [**Context**](#context)
  - [**Project Overview**](#project-overview)
  - [**Version History**](#version-history)
  - [1. **Project Setup**](#1-project-setup)
  - [2. **Project Overview**](#2-project-overview)
  - [3. **Problem Statement**](#3-problem-statement)
  - [4. **Features Dataset**](#4-features-dataset)
  - [5. **Solution Scope**](#5-solution-scope)
  - [6. **Solution Approach**](#6-solution-approach)
  - [7. **Deployment Strategy**](#7-deployment-strategy)
  - [8. **Wireframe Visualization**](#8-wireframe-visualization)

## **Project Overview**

| Field                  | Details                     |
| ---------------------- | --------------------------- |
| **Project Name**       | US Visa Approval Prediction |
| **Prepared By**        | Md. Alahi Almin Tansen      |
| **Revision Number**    | 1.0                         |
| **Last Revision Date** | 02-09-2025                  |

---
[⬆️ Go to Context](#context)

## **Version History**

| Revision | Date       | Changes Made  | Author                 |
| -------- | ---------- | ------------- | ---------------------- |
| 1.0      | 02-09-2025 | Initial Draft | Md. Alahi Almin Tansen |

---
[⬆️ Go to Context](#context)

## 1. **Project Setup**

- **Requirements**
  - Anaconda
  - Git
  - VSCode

---
[⬆️ Go to Context](#context)

## 2. **Project Overview**

- Problem Statement Understanding
- Solution Understanding
- Code Understanding & Walkthrough
- Deployment Understanding

---
[⬆️ Go to Context](#context)

## 3. **Problem Statement**

- Predict US visa approval status.
- Features considered:
  - Continent, Education, Job Experience, Training, Employment, Age, etc.
- Goal: Predict whether an application will be **Approved** or **Rejected**.

---
[⬆️ Go to Context](#context)

## 4. **Features Dataset**

- **Continent** → Asia, Africa, North America, Europe, South America, Oceania
- **Education** → High School, Bachelor’s, Master’s, Doctorate
- **Job Experience** → Yes / No
- **Training Required** → Yes / No
- **Number of Employees** → 15,000 – 40,000
- **Region of Employment** → West, Northeast, South, Midwest, Island
- **Prevailing Wage** → 700 – 70,000
- **Contract Tenure** → Hour / Year / Week / Month
- **Full-Time Position** → Yes / No
- **Age of Company** → 15 – 180

---
[⬆️ Go to Context](#context)

## 5. **Solution Scope**

- Helps US visa applicants evaluate their chances.
- Provides insights on **what features improve approval chances**.

---
[⬆️ Go to Context](#context)

## 6. **Solution Approach**

- **Machine Learning Classification Algorithms**
- Train, Validate, Test on visa application data

---
[⬆️ Go to Context](#context)

## 7. **Deployment Strategy**

- **Docker** → Containerize application
- **Cloud Services** → Host ML model & API
- **Self-hosted Runner** → Automate deployment
- **Workflows** → CI/CD pipeline

---
[⬆️ Go to Context](#context)

## 8. **Wireframe Visualization**

- Project Workflow

  ```mermaid
  flowchart TD
      A[Problem Statement] --> B[Features Dataset]
      B --> C[Solution Approach: ML Classification]
      C --> D[Model Training & Evaluation]
      D --> E[Deployment]
      E --> F[Docker Container]
      E --> G[Cloud Services]
      E --> H[CI/CD Workflows]
      F --> I[End User Access]
      G --> I
      H --> I
  ```

---
[⬆️ Go to Context](#context)

from pathlib import Path
import json
import sys

# Project name
project_name = "us_visa_approval_prediction"

# List of all files and directories to create
files_to_create = [
    # --- Root package ---
    f"{project_name}/__init__.py",

    # --- Components (data & model pipeline steps) ---
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",

    # --- Configuration ---
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/config.py",

    # --- Constants ---
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/constants.py",

    # --- Entities (config/artifacts) ---
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",

    # --- Logging ---
    f"{project_name}/logger/__init__.py",

    # --- Exception ---
    f"{project_name}/exception/__init__.py",

    # --- Pipelines ---
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",

    # --- Utilities ---
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",

    # --- Tests ---
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_data_ingestion.py",
    f"{project_name}/tests/test_data_transformation.py",
    f"{project_name}/tests/test_model_trainer.py",

    # --- Notebooks ---
    f"{project_name}/notebooks/exploration.ipynb",

    # --- Application entry points ---
    f"app.py",
    f"demo.py",

    # --- Project setup ---
    f"requirements.txt",
    f"Dockerfile",
    f".dockerignore",
    f"setup.py",

    # --- Config files ---
    f"{project_name}/config/model.yaml",
    f"{project_name}/config/schema.yaml",

    # --- Data directories with placeholder files ---
    f"{project_name}/data/raw/.gitkeep",
    f"{project_name}/data/interim/.gitkeep",
    f"{project_name}/data/processed/.gitkeep",

    # --- Environment file ---
    f".env",
]

# Create all files and parent directories
for file_path in files_to_create:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and has content
    if path.exists() and path.stat().st_size > 0:
        print(f"{path.name} is already present in {path.parent} and has some content. Skipping creation.")
        continue

    # Create file based on type
    if path.suffix == ".py":
        path.write_text(f"# {path.name}\n")
    elif path.suffix == ".ipynb":
        notebook_content = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        path.write_text(json.dumps(notebook_content, indent=2))
    elif path.name == ".env":
        env_template = """# Environment Variables

# MongoDB connection
DB_NAME=
COLLECTION_NAME=
CONNECTION_URL=
"""
        path.write_text(env_template.strip())
    else:
        path.touch()  # empty file for txt, yaml, Dockerfile, .gitkeep, etc.

# Ensure .gitignore contains rules for data & .env
gitignore_path = Path(".gitignore")
gitignore_rules = """
# Ignore data files but keep folder structure
data/*
!data/**/.gitkeep

# Ignore environment files
.env

# Python common ignores
.vscode
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.db
*.sqlite3
*.log
.env
.venv
*.egg-info/
dist/
build/
.ipynb_checkpoints/
"""

if not gitignore_path.exists() or ".env" not in gitignore_path.read_text():
    gitignore_path.write_text(gitignore_rules.strip())

# Create .python-version with current Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
python_version_path = Path(".python-version")
python_version_path.write_text(python_version)

print(f"Project '{project_name}' structure processed successfully with .env and .python-version ({python_version})!")

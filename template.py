from pathlib import Path
import json

# Project name
project_name = "US Visa Approval Prediction"

# List of all files to create
files_to_create = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/config.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/constants.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logger.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_data_ingestion.py",
    f"{project_name}/tests/test_data_transformation.py",
    f"{project_name}/tests/test_model_trainer.py",
    f"{project_name}/notebooks/exploration.ipynb",
    f"{project_name}/app.py",
    f"{project_name}/demo.py",
    f"{project_name}/requirements.txt",
    f"{project_name}/Dockerfile",
    f"{project_name}/.dockerignore",
    f"{project_name}/setup.py",
    f"{project_name}/config/model.yaml",
    f"{project_name}/config/schema.yaml",
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
    else:
        path.touch()  # empty file for txt, yaml, Dockerfile, etc.

print(f"Project '{project_name}' structure processed successfully!")

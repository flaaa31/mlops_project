# Online Reputation Monitoring

This repository contains the source code for a complete MLOps project aimed at monitoring a company's online reputation through automated sentiment analysis of social media data.
The system implements an automated CI/CD pipeline that handles testing, fine-tuning of a RoBERTa model, and its deployment to the Hugging Face Hub. 
The final application is served via a FastAPI API, featuring an interactive user interface and a real-time monitoring system based on Prometheus and Grafana.

## Local Execution Guide
To run the entire system in a local development environment (or a Codespace), follow these steps.

### Prerequisites
- Python 3.10+
- Docker and Docker Compose installed and running.

### 1. Python Environment Setup
Create and activate a virtual environment, then install the necessary dependencies.

```py
# Create the virtual environment (name may change)
python3 -m venv .venv

# Activate the environment (on Linux/macOS)
source .venv/bin/activate
# .venv/scripts/activate (on Windows)

# Install the optimized dependencies
pip install -r requirements.txt
```

### 2. Launch the FastAPI API
In a terminal, start the Uvicorn application. 

```bash
uvicorn main.app --reload --host 0.0.0.0
```
**It is crucial to use** ```--host 0.0.0.0 ```
to allow Prometheus (running in Docker) to communicate with the API.

The API will be accessible at ```http://127.0.0.1:8000```.

### 3. Launch the Monitoring System
In a second terminal, start the Prometheus and Grafana containers.
```bash
docker-compose up -d
```

- Prometheus will be available at ```http://localhost:9090```.
- Grafana will be available at ```http://localhost:3000``` (default credentials: ```admin``` / ```admin```).
### Project Structure
- ```main.py```: The main file defining the FastAPI application and the user interface.
- ```sentiment_analyzer.py```: Contains the logic for loading the model and performing sentiment analysis.
- ```train.py```: Script for fine-tuning the model and deploying it to the Hugging Face Hub.
- ```tests/```: Contains unit tests for the sentiment analyzer.
- ```.github/workflows/ci-cd.yml```: Defines the CI/CD pipeline using GitHub Actions.
- ```docker-compose.yml```: Configuration to launch Prometheus and Grafana.
- ```prometheus.yml```: Configuration file for Prometheus. 
- ```PROJECT_DOCUMENTATION.md```: Detailed project documentation.
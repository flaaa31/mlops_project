# Project Documentation: MLOps Pipeline for Sentiment Analysis

## 1. Introduction and Objectives  
This document details the design choices, implementations, and results achieved during the "Online Reputation Monitoring" project. The primary objective was to integrate MLOps methodologies to create a robust, automated system for the development, deployment, and monitoring of a sentiment analysis model, intended to track a company's reputation on social media.  
The project was successfully completed across three main phases:
1. Model Implementation: Selection and fine-tuning of a sentiment analysis model.
2. CI/CD Pipeline Creation: Automation of testing, training, and deployment processes.
3. Continuous Deployment and Monitoring: Serving the model via an API and setting up a performance monitoring system.

## 2. Solution Architecture
 The key components are:
 - Source Code (GitHub): The central repository for all code and configurations.
 - CI/CD Pipeline (GitHub Actions): The automation engine that orchestrates the entire workflow.
 - Model Registry (Hugging Face Hub): The platform for versioning and distributing the trained models.
 - Serving API (FastAPI): The web service that exposes the model for consumption.
 - Monitoring Stack (Prometheus & Grafana): The system for collecting and visualizing operational metrics.
 
 ## 3. Design Choices and Implementations
 ### Phase 1: Model Implementation
 #### Model Selection
 The base model chosen was ```cardiffnlp/twitter-roberta-base-sentiment-latest```. This decision was driven by RoBERTa's powerful architecture and the fact that this specific model was pre-trained on a large corpus of tweets. This makes it exceptionally well-suited to understand the informal language, slang, and nuances of social media text.
 
 #### Fine-Tuning (```train.py```)
 To further specialize the model for our specific task, a fine-tuning process was implemented in ```train.py```.
 - Dataset Selection: The initial choice, sentiment140, was discarded due to a RuntimeError caused by recent security updates in the datasets library that block remote script execution. The more modern and robust tweet_eval dataset was chosen as a replacement. It is a standard benchmark for Twitter-based tasks, ensuring high-quality data.
 - Implementation Details:
    - The script uses the datasets library to load the data. To ensure efficient training within the resource constraints of CI/CD runners, a random subset of the data is selected using .shuffle(seed=42).select(range(...)).
    - The core of the fine-tuning process is handled by the transformers library, specifically using the Trainer class, which abstracts away the complex training loop. TrainingArguments are configured to manage hyperparameters like learning rate, number of epochs, and evaluation strategy.
    - Challenge & Solution: A ValueError occurred because load_best_model_at_end=True was incompatible with save_strategy="no". The logic was that the trainer cannot load the best model if it never saves any. The fix was to set save_strategy="epoch" and save_total_limit=1, telling the trainer to save a checkpoint after each epoch but only keep the single best one, thus preventing disk space issues.
    
    ### Phase 2: CI/CD Pipeline Creation
    #### Technology Choice: GitHub Actions
    
    GitHub Actions was selected for its seamless integration with the source code repository, allowing for the definition of complex workflows directly in YAML (.github/workflows/ci-cd.yml).
    
    #### Pipeline Structure
    The pipeline is logically divided into two sequential jobs to ensure robustness:
    1. test Job: Acts as a quality gate. It runs the unit tests defined in tests/test_sentiment.py using pytest. This verifies the application's core logic before proceeding.
    2. train-and-deploy Job: This job is conditionally executed upon the success of the test job, specified by needs: test. It handles the model fine-tuning and deployment. This dependency prevents wasting computational resources on training a model if the underlying code is unstable.
    #### Deployment and Secrets Management
    The trained model is automatically published to the Hugging Face Hub. To manage authentication securely, the pipeline leverages GitHub Secrets. The HF_TOKEN and HF_USERNAME are injected into the workflow as environment variables, which are then read by the train.py script to authenticate and execute trainer.push_to_hub().
    
    ### Phase 3: Continuous Deployment and Monitoring
    #### Serving API with FastAPI (```main.py```)
    The FastAPI framework was chosen to serve the model for several key reasons:
    - High Performance: Built on Starlette and Pydantic, it offers excellent asynchronous performance.
    - Rapid Development: Automatic interactive documentation (via Swagger UI at /docs) and data validation with Pydantic models (SentimentRequest, SentimentResponse) significantly speed up development and testing.
    - Interactive UI: To enhance usability, a complete HTML/CSS/JavaScript user interface was embedded directly into the main.py file. The root endpoint (/) returns an HTMLResponse, providing a user-friendly way to interact with the model without needing to use the /docs page.
    #### Monitoring Stack with Prometheus & Grafana
    An industry-standard monitoring stack was implemented using Prometheus and Grafana.
    - Metrics Exposure: The prometheus-fastapi-instrumentator library was integrated into main.py. Instrumentator().instrument(app).expose(app) automatically creates standard metrics (e.g., http_requests_latency_seconds, http_requests_total) exposed at the /metrics endpoint. A custom metric, sentiment_analysis_predictions_total = Counter(...), was also created to track the count of predictions for each sentiment label.
    - Service Orchestration (docker-compose.yml): A docker-compose.yml file was created to simplify the launch and management of the Prometheus and Grafana services.
    - Networking Challenge & Solution: A critical connection refused error occurred where Prometheus (inside Docker) could not scrape the FastAPI API (running on the host). The root cause was that Uvicorn, by default, binds to localhost (127.0.0.1), rejecting external connections. The solution, now documented in the README.md, was to run Uvicorn with the --host 0.0.0.0 flag. This command binds the application to all available network interfaces, making the /metrics endpoint accessible to Prometheus. The extra_hosts configuration was also added to docker-compose.yml to ensure reliable name resolution from the container to the host.
    - Data Persistence: To ensure Grafana dashboards and Prometheus data survive restarts, Docker Volumes were configured in docker-compose.yml. This makes the monitoring stack stateful and reliable.
    
    ### 4. Final Results
    The project successfully achieved its objectives, resulting in a complete, end-to-end MLOps system.
    - Fine-Tuned Model: A specialized version of RoBERTa for sentiment analysis is versioned and available on the Hugging Face Hub.
    - Automated Pipeline: A fully automated CI/CD workflow ensures that every code change is tested and that the model is consistently retrained and published.
    - Interactive API: The final application provides an intuitive web interface for human users.
    - Monitoring Dashboard: A Grafana dashboard provides real-time insights into both the API's performance (latency, throughput) and the model's predictions (sentiment distribution).

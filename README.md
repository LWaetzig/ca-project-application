# Table of Content

## TL;DR

## Repository Structure

This Repository is structured in two different components.

- [app](./app):
  - contains the Streamlit application
- [ollama](./ollama):
  - contains configuration for the Ollama API client

## Local Setup (using Docker)

> **IMPORTANT**: Before you begin, follow the steps below and ensure you have completed all pre-requisites.

### Pre-Requisites

- Required Software:
  - [python >=3.11](https://www.python.org/downloads/),
  - [Docker & docker-compose](https://docs.docker.com/),
- After cloning the repository
  - download the data from the [Google Drive](https://drive.google.com/drive/folders/1NYCDdsTFyRenn_rzbDNEzH9H_K4w_UmZ?usp=sharing)
  - navigate into the `app` directory
  - create a new folder called `data` and copy all files from the Google Drive into this directory

### Setup & Build Docker Containers

> **IMPORTANT**: Follow the steps listed below the [Remarks before Building](#remarks-before-building) before proceeding with the setup.

- To build and start the Docker containers, run the following command from the `root` directory of the repository:

```bash
docker-compose up --build
```

- The Streamlit application should be available at `http://localhost:8501`.

>**note**: On the first start, it may take a few minutes for the containers to build and start up. Especially for the LLM (Ollama) container, which require additional time to download and set up the language model.

#### Remarks before Building

> **IMPORTANT**: Before building the containers, please read the following remarks carefully.

- Ensure that you have the latest version of Docker and docker-compose installed.
- Consider increasing the resources allocated to Docker (CPU, memory)
  - The default settings in the [docker-compose.yml](./docker-compose.yml) require at least 8 GB of RAM
  - If your system has limited resources, you may need to adjust the services to use less memory (especially for the LLM container)

#### LLM (Ollama) Container Setup

> **TL;DR**: The LLM (Ollama) container is responsible for serving the language model API.

- This container is built using the `Dockerfile` located in the `ollama` directory.
- [Ollama](https://ollama.com/) is a powerful framework for deploying and serving large language models.
- The container exposes the API on port 11434.
- The model used is the `llama3.1:8b` but can (should) be replaced -> see [notes on ollama usage](#notes-on-ollama-usage)

#### Streamlit Application Setup

> **TL;DR**: The Streamlit Application is the user interface for interacting with the data.

- This container is built using the `Dockerfile` located in the `app` directory.
- [Streamlit](https://streamlit.io/) is an open-source app framework for developing quick and interactive data applications for machine learning and data science purposes.
- A detailed description of the application capabilities and features can be found in the [documentation]().

## Notes

### Notes on Ollama Usage

- The Ollama container is designed to serve the language model API efficiently.
- Ensure the model you intend to use suits your system's resources
  - Some Models may require more VRAM than others, so it's important to check the model documentation for details
  - Suitable and lightweight models are:
    - `llama3.1:8b` or llama 3 with less parameters
    - `gemma3:4b` or gemma 3 with less parameters
    - `qwen2.5:7b` or qwen 2.5 with less parameters
  - Large and resource-intensive models are:
    - `gpt-oss`
    - `phi4`
    - all models with more than 8b parameters

## Troubleshooting

- If you encounter issues with the Ollama container, consider the following steps:
  - Check the container logs for any error messages or warnings.
  - Ensure that your system meets the resource requirements for the selected model.
  - If the model fails to load, try using a lighter model or adjusting the container's resource limits.
  - Consult the [Ollama documentation](https://ollama.com/docs) for additional troubleshooting tips.

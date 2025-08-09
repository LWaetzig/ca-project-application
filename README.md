# Table of Content

- [Repository Structure](#repository-structure)
- [Local Setup (using Docker)](#local-setup-using-docker)
  - [Pre-Requisites](#pre-requisites)
  - [Pre-Setup](#pre-setup)
  - [Setup Docker Containers](#setup-docker-containers)
  - [Database Container Setup](#database-container-setup)
    - [Database: Initial Start / Reset](#database-initial-start--reset)

## Repository Structure

This Repo is structured in three different parts.

- [database](./database):
  - docker-container for the postgres database
  - contains scripts that update the database
- [python](./python):
  - includes every python script in different subsections to download & generate data, sorted by execution order
- [app](./app):
  - contains the Streamlit application

## Local Setup (using Docker)

### Pre-Requisites

- Required Software:
  - [python >=3.11](https://www.python.org/downloads/),
  - [yarn](https://yarnpkg.com/),
  - [Docker / docker-compose](https://docs.docker.com/),
  - [node](https://nodejs.org/en/download) - ideally installed via node version manager (nvm)

### Pre-Setup

- run the following commands in the dedicated terminal
  - `yarn` in the `database` directory
  - `sh setup.sh` in the `python` directory

### Setup Docker Containers

- for the initial build: run `docker-compose build` in the `root` folder

#### Database Container Setup

The following steps will guide you through setting up and starting the database

0. Change into the database directory and make sure the database container is running (if not: `docker-compose up -d database`)
1. Upload the schema to the database by executing the following command:

```shell
// run from database folder
yarn run db:update:local
```

2. Change into the python directory
3. Download / Generate Data by executing the following command:

```shell
// run from python folder
sh build.sh
```

> **note:** This script is a pipeline executing all scripts in `python/src`. A detailed description of each script can be found in the [README in src](./python/src/README.md).

#### Streamlit Application Setup

> **note:** In order to be able to use the Streamlit application, you need to complete the setup steps for the database and the Python scripts first.

- no setup required
- run the following command to start the complete application:

```shell
// run from root folder
docker-compose up
```

- the streamlit application should be available at `http://localhost:8501`

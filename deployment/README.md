# PyTupli Deployment

This document outlines the steps to deploy the Pytupli application using Docker Compose for both development and production environments.

## Prerequisites

- Docker and Docker Compose installed.
- Git (for cloning the repository).
- Ports 80, 443 accessible from the Internet (for production only).

## Directory Structure

- `docker_compose/`: Contains Docker Compose files and related scripts.
  - `docker-compose.yaml`: For development.
  - `docker-compose.prod.yaml`: For production.
  - `Dockerfile`: Defines the Pytupli API server image.
  - `init-letsencrypt.sh`: Script to initialize SSL certificates for production.
  - `env.template`: Template for API server environment variables.
  - `env.mongo.template`: Template for MongoDB environment variables.
  - `env.nginx.template`: Template for Nginx environment variables.
- `data/`: Contains persistent data for services, including Nginx configurations and Certbot certificates (once acquired).

## Environment Configuration

Before running the application, you need to set up environment variable files. In the `deployment/docker_compose/` directory:

1.  **API Server Configuration**: Copy `env.template` to `.env` and fill in the required values.
    ```bash
    cp env.template .env
    ```
2.  **MongoDB Configuration**: Copy `env.mongo.template` to `.env.mongo` and update values if desired.
    ```bash
    cp env.mongo.template .env.mongo
    ```
3.  **Nginx Configuration (Production Only)**: Copy `env.nginx.template` to `.env.nginx` and set your `DOMAIN` and `EMAIL` for SSL certificate generation.
    ```bash
    cp env.nginx.template .env.nginx
    ```

## Production Deployment

The production setup includes the API server, MongoDB, Nginx as a reverse proxy, and Certbot for SSL certificate management with Let's Encrypt.

### Services

-   `api_server`: The Pytupli FastAPI application.
-   `mongo`: MongoDB database instance.
-   `nginx`: Nginx reverse proxy for handling HTTP/HTTPS traffic and SSL termination.
-   `certbot`: Manages SSL certificates using Let's Encrypt.

### Steps

1.  Navigate to the `deployment/docker_compose/` directory:
    ```bash
    cd deployment/docker_compose
    ```
2.  Ensure your `.env`, `.env.mongo`, and `.env.nginx` files are configured. Pay special attention to `DOMAIN` in `.env.nginx` for SSL.
3.  **Initialize SSL Certificates (First-time setup):**
    Make the `init-letsencrypt.sh` script executable and run it:
    ```bash
    chmod +x init-letsencrypt.sh
    ./init-letsencrypt.sh
    ```
    This script will:
    - Download recommended TLS parameters.
    - Create a dummy certificate for initial Nginx startup.
    - Start Nginx.
    - Request a Let's Encrypt certificate for your domain.
    - Reload Nginx with the new certificate.
    - Set up automatic certificate renewal.

4.  **Start Production Services:**
    For subsequent starts, use the production Docker Compose file:
    ```bash
    docker compose -f docker-compose.prod.yaml -p pytupli-stack --profile with_mongo up --build -d
    ```
    The `-p pytupli-stack` flag sets a project name to avoid conflicts. The `--profile with_mongo` ensures the `mongo` service defined in `docker-compose.prod.yaml` is started. If you wish to use an external MongoDB instance, you can omit this profile and configure the `MONGO_CONNECTION_STRING` in your `.env` file to point to your external database. The `init-letsencrypt.sh` script uses the `with_mongo` profile by default when starting services.

The application will be accessible via `http://<your_domain>` and `https://<your_domain>`.

### Data Persistence

-   **MongoDB**: Data is persisted in a Docker volume named `mongo_data`.
-   **Nginx**: Configuration is mounted from `../data/nginx`.
-   **Certbot**: Certificates and configuration are mounted from `../data/certbot`.

Ensure these volumes are backed up as needed.

## Development Deployment

The development setup runs the Pytupli API server and a MongoDB database.

### Services

-   `api_server`: The Pytupli FastAPI application.
-   `mongo`: MongoDB database instance.

### Steps

1.  Navigate to the `deployment/docker_compose/` directory:
    ```bash
    cd deployment/docker_compose
    ```
2.  Ensure your `.env` and `.env.mongo` files are configured.
3.  Build and start the services. To include the local MongoDB container, use the `--profile with_mongo` flag:
    ```bash
    docker compose -f docker-compose.yaml --profile with_mongo up --build
    ```
    To run in detached mode, add the `-d` flag:
    ```bash
    docker compose -f docker-compose.yaml --profile with_mongo up --build -d
    ```
    If you wish to use an external MongoDB instance, you can omit the `--profile with_mongo` flag and ensure the `MONGO_CONNECTION_STRING` in your `.env` file points to your external database.

The API server will be accessible at `http://localhost:${PORT}` (default `8080`, as specified in your `.env` or `Dockerfile`).

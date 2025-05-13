# Use a base Python image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create a virtual environment inside the Docker container
RUN poetry config virtualenvs.create false

# Copy the startup script into the container
COPY ./start_server.sh /app/start.sh
#RUN chmod +x /app/start.sh

# Ensure the script has Unix line endings and executable permissions
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix /app/start.sh && \
    chmod +x /app/start.sh && \
    apt-get remove -y dos2unix && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy Poetry configuration files
COPY ./pyproject.toml ./poetry.lock* /app/

# Copy roles.py script over to initialize the database
COPY ./pytupli/server/management/roles.py /app/roles.py

# Copy the `pytupli` module into the container
COPY ./pytupli /app/pytupli

# Install the `pytupli` module using Poetry
RUN poetry install --with server

# Expose the port from the PORT environment variable (default to 8080)
ENV PORT=8080
EXPOSE ${PORT}

# Run the FastAPI app
CMD ["/app/start.sh"]

FROM ollama/ollama:latest

# Install curl for healthchecks and API testing
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the custom entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Set the custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]

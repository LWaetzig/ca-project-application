#!/bin/sh

# Start Ollama server in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama server is ready. Pulling model..."

# Pull the required model
ollama pull qwen2.5vl:7b

echo "Model pulled successfully. Keeping server running..."

# Keep the container running
wait

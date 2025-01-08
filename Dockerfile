FROM python:3.9.13

# Set the working directory
WORKDIR /empathic-agents

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . /empathic-agents

# Expose the port where program
EXPOSE 8765

# Default command to run main.py
CMD ["python", "LLM_Character/webSocketServer.py"]
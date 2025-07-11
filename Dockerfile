FROM python:3.9.13-slim

# Set the working directory
WORKDIR /empathic-agents

# Copy requirements.txt into the container
COPY requirements-small.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements-small.txt

# Copy the project code
COPY . /empathic-agents
RUN pip install --editable .

# Expose the port where program
EXPOSE 8765

# Env key


# Default command to run main.py
CMD ["python", "LLM_Character/app.py"]
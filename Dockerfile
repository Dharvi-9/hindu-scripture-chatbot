# Use official python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list first
COPY requirements.txt .

# Insatll python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . . 

# Export FastAPI port
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]



## TO RUN THE PROJECT ROOT (docker build -t scripture-chatbot .)
## TO START THE CONTAINER( docker run -p 8000:8000 --env-file .env scripture-chatbot) 
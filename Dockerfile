FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos SOLO requirements primero para cachear dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Instalamos dependencias UNA sola vez gracias al cache
RUN pip install --no-cache-dir -r requirements.txt

# Ahora copiamos el proyecto completo
COPY . .

# Exponemos Flask
EXPOSE 5000

CMD ["python", "project/app.py"]

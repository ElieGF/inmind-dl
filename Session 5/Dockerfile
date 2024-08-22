FROM python:3.11.0-slim


RUN mkdir app
WORKDIR /app


COPY simple_requirements.txt .
RUN pip install -r simple_requirements.txt


EXPOSE 8000
COPY . .

CMD ["uvicorn","simple_model:app", ,"0.0.0.0", "-p", "8000"]


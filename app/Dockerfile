FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY main.py /app/
COPY config.toml /app/
COPY credentials.toml /app/
COPY test_images/ /app/test_images/
COPY trained_model/ /app/trained_model/

RUN pip install -r requirements.txt

EXPOSE 8501

RUN mkdir -p ~/.streamlit

RUN cp credentials.toml ~/.streamlit/credentials.toml

RUN cp config.toml ~/.streamlit/config.toml

ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0" ]
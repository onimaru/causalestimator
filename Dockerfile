FROM python:3.8-slim-buster
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
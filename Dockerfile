FROM python:latest

RUN pip install streamlit plotly matplotlib xlrd lxml statsmodels

COPY . /app

WORKDIR ./app

CMD ["streamlit", "run", "covid_app_final.py"]
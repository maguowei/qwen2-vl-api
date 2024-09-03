FROM python:3.12-slim
WORKDIR /app
ENV PIP_INDEX_URL https://mirrors.aliyun.com/pypi/simple/
ENV PIPENV_PYPI_MIRROR https://mirrors.aliyun.com/pypi/simple/
RUN pip install -U pip && pip install poetry

COPY pyproject.toml poetry.* ./
RUN poetry install
COPY . ./

CMD ["poetry run fastapi run"]
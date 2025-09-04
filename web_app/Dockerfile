
FROM python:3.9


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./ /code/web_app


CMD ["fastapi", "run", "web_app/inference_pipeline.py", "--port", "80"]
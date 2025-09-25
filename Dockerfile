# Use an official lightweight Python image as the base
FROM python:3.11

# Set the working directory in the container
WORKDIR /web_app

# Copy the requirements file
COPY ./web_app/requirements.txt /web_app//requirements.txt

# Install the necessary Python packages
RUN pip install --no-cache-dir -r /web_app/requirements.txt

# Copy the application code


# Expose the port the app runs on
EXPOSE 8050


COPY ./web_app  /web_app
COPY ./reports /web_app/reports
COPY ./src  /web_app/src
COPY ./monitoring  /web_app/monitoring
COPY ./models /web_app/models



# Command to run the application
CMD ["python", "app.py"]
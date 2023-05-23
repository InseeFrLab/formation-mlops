FROM inseefrlab/onyxia-python-minimal:py3.10.9

# set current work dir
WORKDIR /formation-mlops

RUN chown -R ${USERNAME}:${GROUPNAME} /formation-mlops

# copy project files to the image
COPY . .

# install all the requirements and import corpus
RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    python -m nltk.downloader stopwords

# launch the unicorn server to run the api
# If you are running your container behind a TLS Termination Proxy (load balancer) like Nginx or Traefik,
# add the option --proxy-headers, this will tell Uvicorn to trust the headers sent by that proxy telling it
# that the application is running behind HTTPS, etc.
CMD ["uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

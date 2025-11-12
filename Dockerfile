FROM inseefrlab/onyxia-python-minimal:py3.12.12
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

# set current work dir
WORKDIR /formation-mlops

# copy project files to the image
COPY --chown=${USERNAME}:${GROUPNAME} . .

# install all the requirements and import corpus
RUN uv sync --frozen && \
    uv run python -m nltk.downloader stopwords

# launch the unicorn server to run the api
EXPOSE 8000
CMD ["uv","run","uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]

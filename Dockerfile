# Reliable container build: micromamba installs fpocket from bioconda so a
# hosted demo (e.g. Hugging Face Spaces "Docker" template) works out of the box.
FROM mambaorg/micromamba:1.5.8

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml
RUN micromamba install -y -n base -f /app/environment.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

EXPOSE 8501
# HF Spaces sets $PORT; default to 8501 locally.
ENV PORT=8501
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

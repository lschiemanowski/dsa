FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    JOBLIB_MULTIPROCESSING=0 \
    LOKY_MAX_CPU_COUNT=1 \
    MALLOC_CONF=background_thread:false \
    ARROW_DEFAULT_MEMORY_POOL=system

RUN groupadd --gid 1000 dsa \
    && useradd --uid 1000 --gid 1000 --no-create-home dsa \
    && python -m pip install --no-cache-dir --only-binary=:all: \
        contourpy==1.3.3 \
        cycler==0.12.1 \
        duckdb==1.5.5 \
        fonttools==4.63.0 \
        joblib==1.5.3 \
        kiwisolver==1.5.0 \
        matplotlib==3.11.1 \
        narwhals==2.24.0 \
        numpy==2.5.1 \
        packaging==26.3 \
        pandas==3.0.5 \
        patsy==1.0.2 \
        pillow==12.3.0 \
        pyarrow==25.0.1 \
        pyparsing==3.3.2 \
        python-dateutil==2.9.0.post0 \
        scikit-learn==1.9.0 \
        scipy==1.18.0 \
        six==1.17.0 \
        statsmodels==0.14.6 \
        threadpoolctl==3.6.0

USER 1000:1000
WORKDIR /workspace
CMD ["/usr/local/bin/python", "-I", "-B", "-"]

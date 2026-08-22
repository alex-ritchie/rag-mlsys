# syntax=docker/dockerfile:1.7
# CPU image: BASE=mlsysbook-rag/python-base:cpu-models ; GPU image: BASE=mlsysbook-rag/python-base:gpu-models
ARG BASE=mlsysbook-rag/python-base:cpu-models
FROM ${BASE}
EXPOSE 8001
ENV EMBEDDER_MODE=cpu
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health').status==200 else 1)"
CMD ["uvicorn", "mlsys_embedder.app:app", "--host", "0.0.0.0", "--port", "8001"]

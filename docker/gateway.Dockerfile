# syntax=docker/dockerfile:1.7
ARG BASE=mlsysbook-rag/python-base:cpu
FROM ${BASE}
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "mlsys_gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

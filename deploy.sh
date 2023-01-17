killall gunicorn
gunicorn api:app -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 --daemon
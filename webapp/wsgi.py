"""WSGI entrypoint. Mostly used for gunicorn. All configuration done via env variables."""
from dfsurvey.app.factory import create_app

app = create_app()

if __name__ == "__main__":
    app.run()

import multiprocessing as mp
import os
import socket
import tempfile
import time
from ctypes import c_char_p, c_uint

from dfsurvey.app.factory import create_app
from dfsurvey.models.factory import init_db
from flask import Flask

from test_utils import TestConfig


def worker_fn(port_value, db_value, config: TestConfig):
    """Worker function for subprocess to run the server.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp.name}"

        # write back newly created db
        db_value.value = config.SQLALCHEMY_DATABASE_URI

        app = create_app(config)
        with app.test_request_context():
            with app.app_context():
                init_db()

                # create a new socket on 0
                # aka tell your OS to give you a free socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", 0))
                sock.listen()

                # tell WERKZEUG dev server to use this socket
                os.environ["WERKZEUG_SERVER_FD"] = str(sock.fileno())

                # Get the assigned socket
                _, port = sock.getsockname()
                port_value.value = port

                app.run(use_reloader=False)


class LiveServer:
    """
    Test case class for running a life server.
    Loosley based on https://github.com/jarus/flask-testing/blob/master/flask_testing/utils.py

    Args:
        timeout (int): Timeout for server start (Default 5s)
        port (int): Port for running the server on (Default 8000).
    """

    def __init__(
        self,
        timeout: int = 5,
        port: int = 8000,
        config: TestConfig | None = None,
    ) -> None:
        self.timeout = timeout
        self.port = port
        self.config = config or TestConfig()

        manager = mp.Manager()
        self.db_uri = manager.Value(c_char_p, "")
        self.port_uri = manager.Value(c_uint, 0)

        # self.__config = config
        self.__manager = manager
        self.__process = None

    def __start_server(self):
        args = (self.port_uri, self.db_uri, self.config)
        self.__process = mp.Process(
            # , self.__config),
            target=worker_fn, args=args,
        )

        self.__process.start()
        start_time = time.time()

        while True:
            elapsed = (time.time() - start_time)
            if elapsed > self.timeout:
                raise RuntimeError(
                    f"Failed to start the server after {self.timeout}s!"
                )

            if self.can_ping_server():
                break

        # set db_uri
        self.config.SQLALCHEMY_DATABASE_URI = self.db_uri.value

    def app(self) -> Flask:
        """Create an application context based on the config.

        This has the newly created tmp db set.
        """
        return create_app(self.config)

    def server_url(self) -> str:
        """Return the server url.
        """
        return f"http://localhost:{self.port}"

    def can_ping_server(self) -> bool:
        """Test if we can ping (connect) to the server.
        """
        if self.port_uri.value == 0:
            return False

        self.port = self.port_uri.value

        # try connecting
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.connect(("localhost", self.port))
        except socket.error:
            success = False
        else:
            success = True
        finally:
            sock.close()

        return success

    def __enter__(self) -> 'LiveServer':
        self.__start_server()
        return self

    def __close(self):
        if self.__process:
            self.__process.terminate()

    def __exit__(self, *_):
        self.__close()

        # just propagate exceptions
        return False

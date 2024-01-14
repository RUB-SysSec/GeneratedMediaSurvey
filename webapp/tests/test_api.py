"""This files contains tests regarding the api module.
"""
import re

from dfsurvey.api_1_0.api import api_bp
from dfsurvey.app.factory import create_app

from test_integration import simulate_clients
from test_utils import IntegrationConfig


def test_stats_basic(n: int = 10):
    """Check that basic statistics work.
    """
    config = IntegrationConfig()
    app = create_app(config)
    app.register_blueprint(api_bp)
    _ = simulate_clients(app, n)

    with app.test_client() as client:
        res = client.get("/stats")
        assert res.status_code == 200

        data = res.data.decode("utf-8")
        amount_of_clients = int(re.search(r"Total.*([0-9]+)", data)[1])
        assert amount_of_clients <= n

        amount_of_finished = int(re.search(r"Finished.*([0-9]+)", data)[1])
        assert amount_of_finished <= n


def test_stats_quotes_laods(n: int = 10):
    """Test that quotes are correctly displayed.
    """
    config = IntegrationConfig()
    app = create_app(config)
    app.register_blueprint(api_bp)
    _ = simulate_clients(app, n)

    with app.test_client() as client:
        res = client.get("/stats")
        assert res.status_code == 200

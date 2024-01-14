from dfsurvey.app.factory import create_app
from dfsurvey.models.user import User

from test_integration import simulate_clients
from test_utils import IntegrationConfig


def test_export():
    """Test that the export function works.
    """
    app = create_app(IntegrationConfig())
    clients = simulate_clients(app, 10)

    with app.app_context():
        assert len(User.query.filter(User.finished).all()) > 0

        for client in clients:
            if not client.finished:
                continue

            user = User.query.get(client.user_id)
            res = user.export()

            # Check all questions exported
            question_answered = {}
            for data in [
                client.questionnaire_picks["demographics"],
                client.questionnaire_picks["presurvey"],
                client.questionnaire_picks["postsurvey"],
            ]:
                question_answered.update(data)

            coded_answers = {
                x["question_id"]: x["option_id"] for x in res["answers"] if "option_id" in x
            }

            for question_id, option in question_answered.items():
                if question_id in coded_answers:
                    assert coded_answers[question_id] == option
                    del coded_answers[question_id]

            assert len(coded_answers) == 0  # all found

            assert client.performed_experiments == len(res["ratings"])
            ratings = {x["index_of_experiment"]: x["rating"]
                       for x in res["ratings"]}

            for index, rating in client.ratings_given.items():
                assert ratings[index] == rating
                del ratings[index]

            assert len(ratings) == 0  # all found

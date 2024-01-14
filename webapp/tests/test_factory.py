from pathlib import PosixPath

from dfsurvey.app.factory import _load_data

from test_utils import LANGUAGES_TO_TEST, MEDIA_TO_TEST, TestConfig

CATEGORIES = ["real", "fake"]


def test_load_data():
    data = _load_data(TestConfig.DATA_DIR)

    for media in MEDIA_TO_TEST:
        assert media in data
        cur_media = data[media]

        for raw_data in cur_media.values():
            for category in CATEGORIES:
                assert category in raw_data
                cat_data = raw_data[category]
                for file_name, path in cat_data.items():
                    assert isinstance(file_name, str)
                    assert isinstance(path, PosixPath)

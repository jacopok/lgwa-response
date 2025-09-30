from lgwa_response import data_path


def test_data_path():
    assert data_path.exists()

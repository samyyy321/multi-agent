def test_fastapi_application_imports():
    """The HTTP application must import before the server can start."""
    from src.main import app

    assert app.title

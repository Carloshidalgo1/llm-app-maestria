import os


def pytest_configure() -> None:
    """Las pruebas unitarias no deben enviar trazas externas a LangSmith."""
    os.environ["LANGSMITH_TRACING"] = "false"

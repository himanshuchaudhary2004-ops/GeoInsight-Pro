# Contributing

1. Create a feature branch.
2. Keep analytical operations reproducible and documented.
3. Do not commit credentials, API keys, private datasets or `.streamlit/secrets.toml`.
4. Run `python -m py_compile app.py` and `pytest -q` before opening a pull request.
5. Update the README or CHANGELOG when adding a user-visible capability.
6. Include scientific limitations when an operation is a preview or approximation.

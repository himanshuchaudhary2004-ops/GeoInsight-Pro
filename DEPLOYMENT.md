# GeoInsight Pro 50 — Deployment Notes

## Streamlit Community Cloud

Use Python **3.12** for this release. The dependency set is intentionally pinned/capped to versions with current binary-wheel support for the Linux cloud environment.

1. Push the repository to GitHub.
2. Deploy `app.py` with Streamlit Community Cloud.
3. In **Advanced settings**, select **Python 3.12** if the deployment dialog asks for a version.
4. Commit any dependency changes to `requirements.txt`; Community Cloud will re-resolve the environment.

If an older failed environment is being reused, redeploy the app rather than changing the Python version in-place.

## GitHub Actions CI
The repository CI uses GitHub Actions with Python 3.12 and installs directly from `requirements.txt`. It does not use Conda or `environment.yml`. If an older workflow in GitHub still runs `conda env update --file environment.yml`, delete that old workflow from `.github/workflows/` and keep only `ci.yml`.

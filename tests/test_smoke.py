from pathlib import Path


def test_app_compiles():
    source = Path(__file__).parents[1] / "app.py"
    compile(source.read_text(encoding="utf-8"), str(source), "exec")


def test_required_repository_files_exist():
    root = Path(__file__).parents[1]
    for name in ["app.py", "requirements.txt", "README.md", "LICENSE", ".gitignore"]:
        assert (root / name).exists(), name

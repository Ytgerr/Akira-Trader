$env:PYTHONPATH = "$PWD"
./.venv/Scripts/activate
uv run src/app/interfaces/chat_cli.py

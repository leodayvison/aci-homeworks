@echo off

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

echo Ambiente configurado!
pause

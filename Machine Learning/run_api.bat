@echo off
setlocal

if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Run setup_and_train.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python app.py

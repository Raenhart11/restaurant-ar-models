@echo off
setlocal

py -3.11 -m venv .venv
if errorlevel 1 goto :error

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error

pip install -r requirements.txt
if errorlevel 1 goto :error

python train_models.py
if errorlevel 1 goto :error

echo.
echo Training completed. Run run_api.bat to start the API.
pause
exit /b 0

:error
echo.
echo Setup or training failed. Review the error above.
pause
exit /b 1

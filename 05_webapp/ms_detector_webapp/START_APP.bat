@echo off
REM MS Lesion Detection Web Application Launcher
REM Starts both backend and frontend servers

echo ======================================================================
echo    MS Lesion Detection Web Application
echo    Production Model: 83.99%% Dice Score
echo ======================================================================
echo.

echo Starting Backend API Server...
start "MS Detection API" cmd /k "cd backend && venv\Scripts\python.exe app.py"

timeout /t 5 /nobreak >nul

echo.
echo Starting Frontend Development Server...
start "MS Detection Frontend" cmd /k "cd frontend && npm start"

echo.
echo ======================================================================
echo Application starting...
echo.
echo Backend API:  http://localhost:5000
echo Frontend UI:  http://localhost:3000
echo.
echo Close the terminal windows to stop the servers
echo ======================================================================
pause

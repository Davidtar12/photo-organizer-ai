@echo off
echo Starting Photo Duplicate Viewer...
echo.
echo [1/3] Activating Python environment...
call ..\photoenv\Scripts\activate.bat

echo [2/3] Starting Flask backend server...
start "Flask Backend" cmd /k "cd backend && python server.py"

timeout /t 3 /nobreak >nul

echo [3/3] Starting React development server...
echo Installing dependencies if needed...
call npm install
call npm run dev

pause

@echo off
echo ============================================================
echo  Photo Duplicate Viewer - First Time Setup
echo ============================================================
echo.

echo Step 1/3: Installing Backend Dependencies (Python)
echo ------------------------------------------------------------
cd backend
if not exist "..\..\photoenv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found!
    echo Please activate photoenv first.
    pause
    exit /b 1
)

echo Using Python from photoenv...
call ..\..\photoenv\Scripts\activate.bat
python -m pip install flask flask-cors pillow
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)
echo ✓ Backend dependencies installed
echo.

echo Step 2/3: Installing Frontend Dependencies (Node.js)
echo ------------------------------------------------------------
cd ..
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: npm not found! Please install Node.js first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

echo Installing React and dependencies...
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)
echo ✓ Frontend dependencies installed
echo.

echo Step 3/3: Checking for duplicates.csv
echo ------------------------------------------------------------
if exist "..\duplicates.csv" (
    echo ✓ duplicates.csv found
) else (
    echo ⚠ WARNING: duplicates.csv not found!
    echo Please run photo-organizer.py first:
    echo    cd "C:\dscodingpython\File organizers"
    echo    photoenv\Scripts\activate
    echo    python photo-organizer.py --root "C:\path\to\photos" --dry-run
    echo.
)

echo.
echo ============================================================
echo  ✓ Setup Complete!
echo ============================================================
echo.
echo To start the application:
echo   • Double-click start.bat
echo   • OR manually run backend and frontend
echo.
echo See SETUP_GUIDE.md for detailed instructions
echo See QUICK_REFERENCE.txt for a quick command reference
echo.
pause

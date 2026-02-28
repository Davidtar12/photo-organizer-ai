@echo off
echo Creating desktop shortcut for Photo Duplicate Viewer...
echo.

set SCRIPT_DIR=%~dp0
set SHORTCUT_TARGET=%SCRIPT_DIR%start.bat
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT_NAME=Photo Duplicate Viewer.lnk

powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%DESKTOP%\%SHORTCUT_NAME%'); $SC.TargetPath = '%SHORTCUT_TARGET%'; $SC.WorkingDirectory = '%SCRIPT_DIR%'; $SC.Description = 'Review and delete photo duplicates'; $SC.Save()"

if %ERRORLEVEL% EQU 0 (
    echo ✓ Shortcut created on your desktop!
    echo   Name: %SHORTCUT_NAME%
    echo.
    echo You can now double-click the desktop icon to start the app.
) else (
    echo ✗ Failed to create shortcut
    echo You can manually run: %SHORTCUT_TARGET%
)

echo.
pause

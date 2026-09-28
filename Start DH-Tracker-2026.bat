@echo off
rem Double-click to start DH-Tracker-2026 (bead localization, tracking and analysis of new data).
rem Leave this window open while you use the explorer; closing it stops the explorer.
setlocal
set "HERE=%~dp0"
set "SELF=%~f0"
set "APP=%HERE%DHPSF_pipeline\explorer\app.py"
set "PORT=8765"

rem Already running? Then just open it.
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo DH-Tracker-2026 is already running; opening it in your browser.
    start "" "http://127.0.0.1:%PORT%"
    exit /b 0
)

rem Installed with the GitHub installer (version.txt holds the installed version)? Then offer updates.
if exist "%HERE%version.txt" if exist "%HERE%install.ps1" call :checkupdate

rem Find a Python that has the packages the pipeline needs.
set "PY="
rem (the dhpsf-tracking-2026 environment in the usual Miniconda / Anaconda locations)
for %%P in ("%USERPROFILE%\miniconda3\envs\dhpsf-tracking-2026\python.exe" "%USERPROFILE%\anaconda3\envs\dhpsf-tracking-2026\python.exe" "%LOCALAPPDATA%\miniconda3\envs\dhpsf-tracking-2026\python.exe" "%LOCALAPPDATA%\anaconda3\envs\dhpsf-tracking-2026\python.exe" "%ProgramData%\miniconda3\envs\dhpsf-tracking-2026\python.exe" "%ProgramData%\anaconda3\envs\dhpsf-tracking-2026\python.exe" "%USERPROFILE%\.conda\envs\dhpsf-tracking-2026\python.exe") do (
    if not defined PY if exist %%P set "PY=%%~P"
)
rem (or wherever conda made it: conda lists its environments in this file)
if not defined PY if exist "%USERPROFILE%\.conda\environments.txt" (
    for /f "usebackq delims=" %%D in ("%USERPROFILE%\.conda\environments.txt") do (
        if not defined PY if /i "%%~nxD"=="dhpsf-tracking-2026" if exist "%%D\python.exe" set "PY=%%D\python.exe"
    )
)
if not defined PY (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY (
            "%%P" -c "import numpy, scipy, tifffile, networkx, torch" >nul 2>&1 && set "PY=%%P"
        )
    )
)
if not defined PY (
    echo.
    echo Could not find a Python with numpy, scipy, tifffile, networkx and torch installed.
    echo First set it up: install Miniconda, then double-click "Setup DH-Tracker-2026.bat" in this folder.
    echo When it says Ready, double-click this file again.
    echo.
    pause
    exit /b 1
)

echo Starting DH-Tracker-2026 with %PY%
echo Your browser will open at http://127.0.0.1:%PORT%
echo Leave this window open while you use the explorer; close it to stop the explorer.
echo An analysis that is running keeps going in the background even if you close this window
echo (the computer is kept from sleeping meanwhile); it appears in the run list when finished.
echo.
"%PY%" "%APP%" --port %PORT% --open-browser
if errorlevel 1 pause
exit /b

rem ---- a newer version on GitHub? (5 s at most; skipped quietly when offline)
:checkupdate
set /p CUR=<"%HERE%version.txt"
set "NEW="
for /f "delims=" %%S in ('powershell -NoProfile -Command "try { (Invoke-RestMethod -UseBasicParsing -TimeoutSec 5 -Headers @{Accept='application/vnd.github.sha'} https://api.github.com/repos/WeissLab/DH-Tracker/commits/main).ToString().Trim() } catch { }" 2^>nul') do set "NEW=%%S"
if not defined NEW exit /b 0
if /i "%NEW%"=="%CUR%" exit /b 0
echo A newer version of DH-Tracker-2026 is available on GitHub.
choice /C YN /T 20 /D N /M "Update now? Your results, calibrations and settings are kept (continues without updating in 20 s)"
if errorlevel 2 exit /b 0
rem (one line: this file is replaced by the update, so cmd must not read further from it; the new copy then starts)
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%install.ps1" -Target "%HERE%." -Update & start "" "%SELF%" & exit

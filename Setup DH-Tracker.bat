@echo off
rem Double-click once on a new computer (after installing Miniconda) to set up DH-Tracker.
rem It creates the Python environment "dh-tracker" with the exact package versions in
rem DHPSF_pipeline\requirements.txt, then runs the tests. Running it again repairs or checks an existing setup.
rem Needs an internet connection (about 1 GB of packages) and takes a few minutes.
setlocal EnableExtensions EnableDelayedExpansion
rem (inside ( ) blocks variables are read as !VAR!: a folder name with brackets, e.g. "Harris_3DPSF (copy)", would break %VAR%)
set "HERE=%~dp0"
if not defined DHPSF_ENV set "DHPSF_ENV=dh-tracker"
set "REQ=%HERE%DHPSF_pipeline\requirements.txt"
set "LOG=%HERE%setup_log.txt"
echo DH-Tracker setup: the Python environment "%DHPSF_ENV%".
echo A log is written to %LOG%
echo.
echo DH-Tracker setup %DATE% %TIME% > "%LOG%"

rem ---- find conda (Miniconda or Anaconda)
set "CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA=%CONDA_EXE%"
for %%P in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
    if not defined CONDA if exist "%%~P\Scripts\conda.exe" set "CONDA=%%~P\Scripts\conda.exe"
)
if not defined CONDA (
    for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
rem conda lists every install and environment it made in this file (covers custom install folders)
if not defined CONDA if exist "%USERPROFILE%\.conda\environments.txt" (
    for /f "usebackq delims=" %%D in ("%USERPROFILE%\.conda\environments.txt") do (
        rem base installs only: an environment (...\envs\name) may contain its own conda.exe
        set "D=%%D"
        if not defined CONDA if "!D:\envs\=!"=="!D!" if exist "%%D\Scripts\conda.exe" set "CONDA=%%D\Scripts\conda.exe"
    )
)
if not defined CONDA (
    echo Could not find conda. Install Miniconda first ^(free, from the Anaconda website; default options^),
    echo then double-click this file again.
    echo.
    pause
    exit /b 1
)
echo Using conda: %CONDA%
echo conda: %CONDA% >> "%LOG%"

rem ---- create the environment unless it exists (conda-forge only: no extra terms to accept).
rem      Never "conda create" an existing name: with -y conda would delete and recreate it.
set "PY="
set "EXISTS="
"%CONDA%" env list 2>nul | findstr /R /C:"^%DHPSF_ENV%  *" /C:"[\\/]%DHPSF_ENV%$" >nul && set "EXISTS=1"
call :findpy
if defined PY (
    echo The environment already exists: !PY!
    echo Checking its packages...
) else if defined EXISTS (
    echo The environment "%DHPSF_ENV%" exists but its Python could not be started.
    echo Remove it in an Anaconda Prompt with:  conda env remove -n %DHPSF_ENV%
    echo then double-click this file again.
    echo.
    pause
    exit /b 1
) else (
    echo Creating the environment ^(Python 3.11^)...
    "!CONDA!" create -y -n "!DHPSF_ENV!" -c conda-forge --override-channels python=3.11 pip >> "!LOG!" 2>&1
    call :findpy
)
if not defined PY (
    echo.
    echo Creating the environment failed. See !LOG!
    echo.
    pause
    exit /b 1
)

rem ---- the exact packages (PyTorch CPU build)
echo Installing the packages ^(about 1 GB on the first run; a few minutes^)...
"%PY%" -m pip install --disable-pip-version-check -r "%REQ%" --extra-index-url https://download.pytorch.org/whl/cpu >> "%LOG%" 2>&1
if errorlevel 1 (
    echo.
    echo Installing the packages failed ^(internet connection?^). See !LOG!
    echo.
    pause
    exit /b 1
)

rem ---- check that everything works
echo Running the tests ^(about 1-2 minutes^)...
pushd "%HERE%DHPSF_pipeline"
"%PY%" -W ignore -m unittest test_pipeline test_track_analysis explorer.test_explorer >> "%LOG%" 2>&1
set "TESTS=%ERRORLEVEL%"
popd
if not "%TESTS%"=="0" (
    echo.
    echo Some tests failed. The program may still work, but please send !LOG! to the developers.
    echo.
    pause
    exit /b 1
)

rem ---- the user's folders (asked once; remembered in the user profile, changeable in the app: menu, Folders)
echo.
echo Two folder dialogs follow ^(they may open behind this window^):
"%PY%" "%HERE%DHPSF_pipeline\user_settings.py" --choose
"%PY%" "%HERE%DHPSF_pipeline\user_settings.py" >> "%LOG%" 2>&1

echo.
echo Ready. Double-click "Start DH-Tracker.bat" to use the program.
echo ^(DH-Tracker guide.mp4 shows a whole session.^)
echo Setup finished OK >> "%LOG%"
echo.
pause
exit /b 0

rem ---- the environment's python.exe, wherever conda put it (envs folder or %USERPROFILE%\.conda\envs)
:findpy
set "PY="
for /f "delims=" %%E in ('call "%CONDA%" run -n "%DHPSF_ENV%" python -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%E"
if defined PY if not exist "%PY%" set "PY="
exit /b 0

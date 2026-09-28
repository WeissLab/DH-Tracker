@echo off
rem DH-Tracker installer: download this one file from https://github.com/WeissLab/DH-Tracker and double-click it.
rem It fetches the newest version of the installer (install.ps1) from GitHub and runs it: you choose the install
rem folder, the program is downloaded there, Miniconda is offered if missing, and the setup runs. Needs an internet connection (about 1 GB of Python packages the first time).
echo DH-Tracker installer (from github.com/WeissLab/DH-Tracker)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; & ([scriptblock]::Create((Invoke-RestMethod -UseBasicParsing https://raw.githubusercontent.com/WeissLab/DH-Tracker/main/install.ps1)))"
echo.
pause

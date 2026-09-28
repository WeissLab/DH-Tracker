# DH-Tracker installer / updater (Windows).
#
# New computer: double-click "Install DH-Tracker.bat" (from the GitHub page), or in PowerShell:
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/WeissLab/DH-Tracker/main/install.ps1)))
#
# 1. asks where to install (default Documents\DH-Tracker)
# 2. downloads the newest version from github.com/WeissLab/DH-Tracker and unzips it there
# 3. offers to install Miniconda (winget) if there is no conda
# 4. runs "Setup DH-Tracker.bat" (Python environment, tests, your movies and results folders)
# 5. opens the install folder, which holds "Start DH-Tracker.bat"
#
# Update (Start DH-Tracker.bat offers it when GitHub has a newer version):
#   install.ps1 -Target <install folder> -Update
# replaces the program files only: results, calibration caches and settings are kept.
param(
    [string]$Target = "",
    [switch]$Update
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"          # Invoke-WebRequest is very slow with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Repo = "WeissLab/DH-Tracker"
$Name = "DH-Tracker"

function Say($msg) { Write-Host $msg }
function Fail($msg) {
    Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red
    try { Read-Host "Press Enter to close" | Out-Null } catch { }
    exit 1
}

function Choose-Folder($default) {
    Say "Choose the folder to install $Name in (a folder dialog; it may open behind this window)."
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $d = New-Object System.Windows.Forms.FolderBrowserDialog
        $d.Description = "$Name`: install folder (a '$Name' folder is created inside the folder you pick)"
        $d.SelectedPath = [Environment]::GetFolderPath("MyDocuments")
        $d.ShowNewFolderButton = $true
        $top = New-Object System.Windows.Forms.Form -Property @{TopMost = $true}
        if ($d.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { return (Join-Path $d.SelectedPath $Name) }
    } catch { }
    $a = Read-Host "Install folder [$default]"
    if ($a) { return $a } else { return $default }
}

# ---- which version: the newest commit on main
try {
    $sha = (Invoke-RestMethod -UseBasicParsing -TimeoutSec 20 -Headers @{Accept = "application/vnd.github.sha"} `
        "https://api.github.com/repos/$Repo/commits/main").ToString().Trim()
} catch { Fail "Could not reach GitHub ($($_.Exception.Message)). Check the internet connection." }
if ($sha -notmatch '^[0-9a-f]{40}$') { Fail "Unexpected answer from GitHub: $sha" }

if (-not $Target) {
    $Target = Choose-Folder (Join-Path ([Environment]::GetFolderPath("MyDocuments")) $Name)
}
$Target = [IO.Path]::GetFullPath($Target)
$verFile = Join-Path $Target "version.txt"
if ($Update -and (Test-Path $verFile) -and ((Get-Content $verFile -Raw).Trim() -eq $sha)) {
    Say "$Name is up to date ($($sha.Substring(0,7)))."; exit 0
}
Say ""
Say "$(if ($Update) {'Updating'} else {'Installing'}) $Name ($($sha.Substring(0,7))) in $Target"

# ---- download and unpack (into a temporary folder first)
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("dhtracker_" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $zip = Join-Path $tmp "src.zip"
    Say "Downloading from github.com/$Repo ..."
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 300 "https://github.com/$Repo/archive/$sha.zip" -OutFile $zip }
    catch { Fail "Download failed ($($_.Exception.Message))." }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
    if (-not $src -or -not (Test-Path (Join-Path $src.FullName "DHPSF_pipeline\analyze.py"))) { Fail "The download does not look like $Name." }
    $oldReq = Join-Path $Target "DHPSF_pipeline\requirements.txt"
    $oldHash = if (Test-Path $oldReq) { (Get-FileHash $oldReq).Hash } else { "" }
    New-Item -ItemType Directory -Force $Target | Out-Null
    # copy over the program files; results (runs\), calibration caches and anything else already there are kept
    robocopy $src.FullName $Target /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "Copying the files to $Target failed (robocopy code $LASTEXITCODE)." }
    Set-Content -Path $verFile -Value $sha -Encoding ascii
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
$newHash = (Get-FileHash (Join-Path $Target "DHPSF_pipeline\requirements.txt")).Hash

# ---- conda: offer Miniconda if there is none
function Find-Conda {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    foreach ($d in "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3",
                   "$env:LOCALAPPDATA\anaconda3", "$env:ProgramData\miniconda3", "$env:ProgramData\anaconda3") {
        if (Test-Path "$d\Scripts\conda.exe") { return "$d\Scripts\conda.exe" }
    }
    $c = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    # conda lists every install and environment it made here (covers custom install folders)
    $list = Join-Path $env:USERPROFILE ".conda\environments.txt"
    if (Test-Path $list) {
        # base installs only: an environment (…\envs\name) may contain its own conda.exe
        foreach ($d in Get-Content $list) {
            if ($d -and $d -notmatch '\\envs\\' -and (Test-Path "$d\Scripts\conda.exe")) { return "$d\Scripts\conda.exe" }
        }
    }
    return $null
}
if (-not (Find-Conda)) {
    Say ""
    Say "$Name needs Miniconda (free) for its Python packages, and it is not installed."
    $a = Read-Host "Install Miniconda now with winget? [Y/n]"
    if ($a -notmatch '^[nN]') {
        winget install --id Anaconda.Miniconda3 -e --source winget --accept-package-agreements --accept-source-agreements
        if (-not (Find-Conda)) { Fail "Miniconda was not found after installing it. Install it from the Anaconda website, then run 'Setup $Name.bat' in $Target." }
    } else {
        Fail "Install Miniconda (from the Anaconda website), then run 'Setup $Name.bat' in $Target."
    }
}

# ---- set up (always on a new install; on an update only when the packages changed)
$setup = Join-Path $Target "Setup $Name.bat"
if (-not $Update -or $oldHash -ne $newHash) {
    Say ""
    Say "Running the setup (Python environment, tests, your folders)..."
    & cmd.exe /c "`"$setup`""
    if ($LASTEXITCODE -ne 0) { Fail "The setup did not finish; see setup_log.txt in $Target." }
}

Say ""
Say "$Name $(if ($Update) {'updated'} else {'installed'}) in $Target ($($sha.Substring(0,7)))."
if (-not $Update) {
    Say "To use it, double-click 'Start $Name.bat' in that folder (opening it now)."
    try { Start-Process explorer.exe -ArgumentList "`"$Target`"" } catch { }
}

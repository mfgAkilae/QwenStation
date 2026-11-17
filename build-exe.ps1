# Builds Qwen-Station.exe (one file, no console window) with PyInstaller.
#   .\build-exe.ps1                 build into .\dist
#   .\build-exe.ps1 -Release        build, then write dist\Qwen-Station-Setup-<version>.exe with a SHA256 file
#   .\build-exe.ps1 -Install -InstallDir <folder>   build and copy the exe into a hand-built engine folder (legacy mode)
#   .\build-exe.ps1 -SkipTests      skip the unit tests before building
param(
    [switch]$Release,
    [switch]$Install,
    [switch]$SkipTests,
    [string]$InstallDir = '',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$build = Join-Path $PSScriptRoot '.build'
$venvPython = Join-Path $build 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    # A private venv: the build never touches the global Python packages.
    & $Python -m venv (Join-Path $build 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the build venv.' }
}
& $venvPython -m pip install --quiet --disable-pip-version-check 'pyinstaller>=6.10,<7'
if ($LASTEXITCODE -ne 0) { throw 'Could not install PyInstaller.' }

if (-not $SkipTests) {
    & $venvPython -m unittest discover -s tests
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; no exe built.' }
}

$icon = Join-Path $build 'qwen-station.ico'
& $venvPython make_icon.py $icon
if ($LASTEXITCODE -ne 0) { throw 'Could not render the icon.' }

# The engine bundle goes in with LF line endings, whatever the checkout did to them: the scripts run in Linux.
$engine = Join-Path $build 'engine'
if (Test-Path $engine) { Remove-Item $engine -Recurse -Force }
Copy-Item (Join-Path $PSScriptRoot 'qwen_station\engine') $engine -Recurse
Get-ChildItem $engine -Recurse -File -Include *.sh, *.py, *.env | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName)
    [IO.File]::WriteAllText($_.FullName, $text.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))
}
Get-ChildItem $engine -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

& $venvPython -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name 'Qwen-Station' --icon $icon `
    --add-data "$(Join-Path $PSScriptRoot 'qwen_station\dashboard.html');qwen_station" `
    --add-data "$engine;qwen_station\engine" `
    --add-data "$(Join-Path $PSScriptRoot 'qwen_station\setup-qwen-lan.ps1');qwen_station" `
    --distpath (Join-Path $PSScriptRoot 'dist') `
    --workpath (Join-Path $build 'work') --specpath $build `
    run-station.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

$exe = Join-Path $PSScriptRoot 'dist\Qwen-Station.exe'
Write-Host "Built: $exe ($([math]::Round((Get-Item $exe).Length / 1MB, 1)) MB)"
if ($Install) {
    if (-not $InstallDir -or -not (Test-Path (Join-Path $InstallDir 'run-variant-logged.sh'))) {
        throw "-InstallDir must name the engine folder (run-variant-logged.sh is missing in '$InstallDir')."
    }
    Copy-Item $exe (Join-Path $InstallDir 'Qwen-Station.exe') -Force
    Write-Host "Installed: $(Join-Path $InstallDir 'Qwen-Station.exe')"
}
if ($Release) {
    $version = (& $venvPython -c "import qwen_station; print(qwen_station.__version__)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $version) { throw 'Could not read the version.' }
    # The exe is installer and app in one, so the release is this single file: nothing to unpack.
    # Started from anywhere, it copies itself to %LOCALAPPDATA%\Programs\Qwen Station under its regular name.
    $setup = Join-Path $PSScriptRoot "dist\Qwen-Station-Setup-$version.exe"
    Copy-Item $exe $setup -Force
    $hash = (Get-FileHash $setup -Algorithm SHA256).Hash.ToLower()
    Set-Content -Path "$setup.sha256" -Value "$hash  $(Split-Path $setup -Leaf)" -Encoding ascii
    Write-Host "Release: $setup ($([math]::Round((Get-Item $setup).Length / 1MB, 1)) MB)"
    Write-Host "SHA256:  $hash"
}

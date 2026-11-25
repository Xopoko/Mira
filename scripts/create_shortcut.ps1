$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\pythonw.exe'
$launch = Join-Path $root 'launch.pyw'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create the Python environment first. See README.md for setup instructions.' }
$desktop = [Environment]::GetFolderPath('Desktop')
$path = Join-Path $desktop 'Mira.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($path)
$shortcut.TargetPath = $python
$shortcut.Arguments = '"' + $launch + '"'
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'Control Windows with gaze and voice'
$shortcut.IconLocation = (Join-Path $root 'context_operator\assets\icon.ico') + ',0'
$shortcut.WindowStyle = 1
$shortcut.Save()
Write-Output ('Shortcut created: ' + $path)

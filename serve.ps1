param(
    [int]$Port = 8080
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
python -m http.server $Port

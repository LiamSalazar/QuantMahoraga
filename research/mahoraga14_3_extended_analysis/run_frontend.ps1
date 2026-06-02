$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $root "frontend")

$nodeBin = "C:\Users\salaz\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
$node = Join-Path $nodeBin "node.exe"
$npmCli = Join-Path $root "outputs\cache\npm-cli\package\bin\npm-cli.js"

if (-not $env:VITE_API_BASE) {
  $env:VITE_API_BASE = "http://127.0.0.1:8000"
}

if (Test-Path -LiteralPath $npmCli) {
  $env:PATH = "$nodeBin;$env:PATH"
  & $node $npmCli install
  & $node $npmCli run dev
} else {
  npm install
  npm run dev
}

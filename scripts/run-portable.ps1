$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$serving = Join-Path $project 'serving'

if (-not (Test-Path (Join-Path $serving 'manifest.json'))) {
    throw 'Portable serving bundle is missing. Re-export the app with scripts/build_serving_bundle.py --portable.'
}

$env:DELHI_AIRCAST_DATA_ROOT = $serving
$env:PYTHONPATH = Join-Path $project 'src'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Install Node.js 22 or newer first: https://nodejs.org/'
}

Push-Location $project
try {
    uv lock
    if ($LASTEXITCODE -ne 0) { throw 'uv lock failed.' }
    uv sync --locked --no-dev
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
}
finally {
    Pop-Location
}

$api = Start-Process -FilePath 'uv' -ArgumentList @('run', '--locked', '--no-dev', 'uvicorn', 'delhi_aircast.api:app', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory $project -PassThru -WindowStyle Hidden
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        try {
            Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        }
        catch {
            if ($api.HasExited) { throw 'Forecast API exited during startup.' }
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) { throw 'Forecast API did not become ready within three minutes.' }

    $env:AIRCAST_API_URL = 'http://127.0.0.1:8000'
    Push-Location (Join-Path $project 'aircast-web')
    try {
        if (-not (Test-Path 'node_modules')) { npm ci }
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        $buildMarker = Join-Path (Get-Location) '.next/BUILD_ID'
        $needsBuild = -not (Test-Path $buildMarker)
        if (-not $needsBuild) {
            $buildTime = (Get-Item $buildMarker).LastWriteTimeUtc
            $needsBuild = [bool](Get-ChildItem app -Recurse -File | Where-Object { $_.LastWriteTimeUtc -gt $buildTime } | Select-Object -First 1)
        }
        if ($needsBuild) {
            npm run build
            if ($LASTEXITCODE -ne 0) { throw 'Next.js production build failed.' }
        }
        npm run start -- --hostname 127.0.0.1
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}

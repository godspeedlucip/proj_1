param(
    [Parameter(Mandatory = $true)][string]$Python310,
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [string]$VenvDir = 'tmp/phase3_nnunet/venv'
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($WorkspaceRoot)
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venv = [IO.Path]::GetFullPath((Join-Path $root $VenvDir))
$contained = $venv.StartsWith($root.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar,
                              [StringComparison]::OrdinalIgnoreCase)
if (-not $contained -or $venv -eq $root -or $venv -eq $source) {
    throw "Virtual environment path must be inside the workspace and distinct from source: $venv"
}
if (-not (Test-Path -LiteralPath $Python310 -PathType Leaf)) {
    throw "Python 3.10 executable not found: $Python310"
}
$version = & $Python310 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne '3.10') {
    throw "Python 3.10 is required; got $version"
}
$venvPython = Join-Path $venv 'Scripts/python.exe'
if (Test-Path -LiteralPath $venv) {
    if (-not (Test-Path -LiteralPath (Join-Path $venv 'pyvenv.cfg') -PathType Leaf) -or
        -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Existing path is not a Python virtual environment: $venv"
    }
} else {
    # Reuse the immutable CUDA torch/numpy installation from the pinned Python base.
    & $Python310 -m venv --system-site-packages $venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$torchVersion = & $venvPython -c 'import torch; print(torch.__version__)'
if ($LASTEXITCODE -ne 0 -or $torchVersion.Trim() -ne '2.3.1+cu118') {
    & $venvPython -m pip install --index-url 'https://download.pytorch.org/whl/cu118' 'torch==2.3.1+cu118'
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
}
& $venvPython -m pip install -c (Join-Path $PSScriptRoot 'constraints-phase3.txt') -e $source 'monai==1.3.0' 'numpy==1.26.4' 'pytest>=8,<9' tomli
if ($LASTEXITCODE -ne 0) { throw 'Phase 3 dependency installation failed' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Phase 3 pip check failed' }
$env:nnUNet_compile = 'false'
$env:nnUNet_n_proc_DA = '0'
& $venvPython -c 'from pathlib import Path; from nnunetv2.phase3.runtime import verify_runtime; import json,sys; print(json.dumps(verify_runtime(Path(sys.argv[1])), indent=2))' $source
if ($LASTEXITCODE -ne 0) { throw 'Phase 3 import verification failed' }
Write-Output "Phase 3 environment ready: $venvPython"

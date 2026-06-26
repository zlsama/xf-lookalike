# Publish to GitHub (requires gh login first)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/publish_github.ps1

$ErrorActionPreference = "Stop"
$Gh = "E:\miniconda\Library\bin\gh.exe"
$RepoName = "xf-lookalike"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $Root

Write-Host "Checking gh auth..."
& $Gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "GitHub CLI is not logged in. Run:"
    Write-Host "  E:\miniconda\Library\bin\gh.exe auth login --hostname github.com --git-protocol ssh --web"
    Write-Host "Then re-run this script after browser authorization."
    exit 1
}

Write-Host "Creating GitHub repo: $RepoName ..."
& $Gh repo create $RepoName --public --source=. --remote=origin --push --description "iFLYTEK Lookalike-AC competition pipeline" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repo may already exist, trying push only..."
    git remote remove origin 2>$null
    $user = (& $Gh api user -q .login)
    git remote add origin "git@github.com:${user}/${RepoName}.git"
    git push -u origin main
}

Write-Host "Done. Repo URL:"
& $Gh repo view --json url -q .url

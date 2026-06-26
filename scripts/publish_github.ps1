# 发布到 GitHub（需先完成 gh 登录）
# 用法: powershell -ExecutionPolicy Bypass -File scripts/publish_github.ps1

$ErrorActionPreference = "Stop"
$Gh = "E:\miniconda\Library\bin\gh.exe"
$RepoName = "xf-lookalike"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $Root

Write-Host "Checking gh auth..."
& $Gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "GitHub CLI 未登录。请先运行:"
    Write-Host "  E:\miniconda\Library\bin\gh.exe auth login --hostname github.com --git-protocol ssh --web"
    Write-Host "然后在浏览器完成授权，再重新运行本脚本。"
    exit 1
}

Write-Host "Creating GitHub repo: $RepoName (private by default, use --public to make public)..."
& $Gh repo create $RepoName --source=. --remote=origin --push --description "iFLYTEK Lookalike-AC competition pipeline" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repo may already exist, trying push only..."
    git remote remove origin 2>$null
    $user = (& $Gh api user -q .login)
    git remote add origin "git@github.com:${user}/${RepoName}.git"
    git push -u origin main
}

Write-Host "Done. Repo URL:"
& $Gh repo view --web 2>$null
& $Gh repo view --json url -q .url

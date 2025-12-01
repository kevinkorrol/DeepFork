# Set strict mode (equivalent to set -uo pipefail)
$ErrorActionPreference = "Stop"

# Use Get-Location to ensure paths are absolute before joining, preventing issues
$TARGET_DIR = ".\data\temp"
$BASE_URL = "https://www.pgnmentor.com/files.html"
$HTML_TMP = [System.IO.Path]::GetTempFileName()

# Function to get the final ZIP URL
function Resolve-Url($link) {
    if ($link -match "^http") {
        return $link
    } else {
        return "https://www.pgnmentor.com/$link"
    }
}

# Function to create the progress bar string
function Get-ProgressBar {
    param(
        [int]$count,
        [int]$total,
        [string]$fname
    )

    $progress = [int](($count * 100) / $total)
    $bar_len = [int]($progress / 2) # 50 chars bar
    $empty_len = 50 - $bar_len

    # Create the progress bar using string multiplication (PowerShell's '*' operator for strings)
    $bar = "#" * $bar_len
    $empty = "-" * $empty_len

    # Use Write-Host with -NoNewline and \r for dynamic updating
    Write-Host "`r[$bar$empty] $progress% ($count/$total) Downloading $fname" -NoNewline
}

# Create directory (-Force is equivalent to mkdir -p)
New-Item -ItemType Directory -Force -Path $TARGET_DIR | Out-Null

Write-Host "Fetching file list from $BASE_URL..."
# curl is an alias for Invoke-WebRequest in PowerShell; -OutFile is -o
Invoke-WebRequest -Uri $BASE_URL -OutFile $HTML_TMP -ErrorAction Stop

Write-Host "Parsing ZIP links..."

# Use Select-String with a RegEx to find links
# The lookbehind (?<=href=") ensures we only capture the link path after the href=
$links = Get-Content $HTML_TMP | Select-String -Pattern 'href="(?<link>[^"]*players/[^"]+\.zip)' -AllMatches |
    ForEach-Object { $_.Matches.Groups["link"].Value } |
    Sort-Object -Unique

if ($links.Count -eq 0) {
    Write-Host "No .zip links found — site format may have changed."
    Remove-Item $HTML_TMP -Force
    exit 1
}

Write-Host "Found $($links.Count) ZIP files. Downloading and extracting..."

$total = $links.Count
$count = 0

foreach ($link in $links) {
    $count++

    if ($count -eq 5) {
        break
    }

    $url = Resolve-Url $link

    # Get the filename (equivalent to basename)
    $fname = [System.IO.Path]::GetFileName($link)
    $dl_path = Join-Path $TARGET_DIR $fname

    # Display progress
    Get-ProgressBar -count $count -total $total -fname $fname

    # Download
    Invoke-WebRequest -Uri $url -OutFile $dl_path -ErrorAction Stop

    unzip -qq -j $dl_path -d $TARGET_DIR | Out-Null

    # Remove the zip file
    Remove-Item $dl_path -Force
}

# Execute the Python function
python3 -c "from src.data_preprocessing import filter_games; filter_games()"

# Clean up temporary data directory and HTML file
Remove-Item $TARGET_DIR -Recurse -Force
Remove-Item $HTML_TMP -Force

Write-Host "`nAll files downloaded and extracted."
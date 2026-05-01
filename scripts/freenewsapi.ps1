# probe_freenewsapi_one_outlet.ps1

chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ErrorActionPreference = "Stop"

$ApiKey = $env:news_triage_codex_app
if (-not $ApiKey) {
  throw "Missing env var news_triage_codex_app"
}

$BaseUrl = "https://api.freenewsapi.io"
$Headers = @{ "x-api-key" = $ApiKey }

# Change these two lines
$OutletName = "The Guardian"
$Domain = "theguardian.com"

$Queries = @(
  "Iran United States casualties death toll",
  "Iran U.S. death toll killed injured",
  "U.S.-Iran war casualties fatalities",
  "Iran United States official death toll",
  "Iran war killed injured United States",
  "Iran missile strike killed injured"
)

$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "freenewsapi_one_outlet_$($OutletName.Replace(' ', '_'))_$Timestamp.log"

function Write-Log {
  param([string]$Text)
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Text
  Write-Host $line
  Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Invoke-FreeNewsGet {
  param(
    [string]$Path,
    [hashtable]$Params
  )

  Start-Sleep -Milliseconds 650

  Invoke-RestMethod `
    -Method Get `
    -Uri "$BaseUrl$Path" `
    -Headers $Headers `
    -Body $Params
}

Write-Log "Probe started"
Write-Log "Outlet=$OutletName Domain=$Domain"
Write-Log "Log file=$LogFile"

Write-Log ""
Write-Log "Looking up publisher UUID..."

$publisherResponse = Invoke-FreeNewsGet `
  -Path "/v1/publishers" `
  -Params @{ q = $OutletName }

foreach ($publisher in $publisherResponse.data) {
  Write-Log ("publisher candidate uuid={0} name={1} domain={2} country={3}" -f `
    $publisher.uuid, $publisher.name, $publisher.domain, $publisher.country)
}

$publisher = $publisherResponse.data |
  Where-Object { $_.domain -and $_.domain.ToLower() -eq $Domain.ToLower() } |
  Select-Object -First 1

if (-not $publisher) {
  $publisher = $publisherResponse.data |
    Where-Object { $_.domain -and ($_.domain.ToLower().Contains($Domain.ToLower()) -or $Domain.ToLower().Contains($_.domain.ToLower())) } |
    Select-Object -First 1
}

if (-not $publisher) {
  $publisher = $publisherResponse.data |
    Where-Object { $_.name -and $_.name.ToLower().Contains($OutletName.ToLower()) } |
    Select-Object -First 1
}

if (-not $publisher) {
  throw "No publisher UUID found for $OutletName"
}

Write-Log ""
Write-Log "Selected publisher name=$($publisher.name) uuid=$($publisher.uuid) domain=$($publisher.domain)"

foreach ($query in $Queries) {
  Write-Log ""
  Write-Log "================ QUERY: $query ================"

  $cursor = $null
  for ($page = 1; $page -le 3; $page++) {
    $params = @{
      q = $query
      language = "en"
      publisher_uuid = $publisher.uuid
      order_by = "archive"
      page_size = 100
    }

    if ($cursor) {
      $params.cursor = $cursor
    }

    Write-Log "Request page=$page params=$($params | ConvertTo-Json -Compress)"

    $response = Invoke-FreeNewsGet -Path "/v1/news" -Params $params

    Write-Log ("meta page_size={0} returned={1} has_more={2}" -f `
      $response.meta.page_size, $response.meta.returned, $response.meta.has_more)

    Write-Log "filters=$($response.meta.filters | ConvertTo-Json -Compress)"

    if (-not $response.data -or $response.data.Count -eq 0) {
      Write-Log "No articles returned."
      break
    }

    foreach ($article in $response.data) {
      Write-Log ("uuid={0} | publisher={1} | date={2} | title={3}" -f `
        $article.uuid, $article.publisher, $article.published_at, $article.title)
    }

    if (-not $response.meta.has_more -or -not $response.meta.next_cursor) {
      break
    }

    $cursor = $response.meta.next_cursor
  }
}

Write-Log ""
Write-Log "Probe finished"
Write-Log "Log file=$LogFile"

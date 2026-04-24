if ([string]::IsNullOrWhiteSpace($env:NEWS_AGENT_KEY)) {
    $userKey = [System.Environment]::GetEnvironmentVariable("NEWS_AGENT_KEY", "User")
    if (-not [string]::IsNullOrWhiteSpace($userKey)) {
        $env:NEWS_AGENT_KEY = $userKey
    }
}

if ([string]::IsNullOrWhiteSpace($env:NEWS_AGENT_KEY)) {
    throw "NEWS_AGENT_KEY is not set in the current shell or your Windows user environment."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

.\.venv\Scripts\adk web --port 8000 --no-reload adk_agents


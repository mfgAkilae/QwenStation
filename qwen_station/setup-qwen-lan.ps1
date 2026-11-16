param(
    [switch]$Check,
    [switch]$Remove,
    [ValidateRange(1, 65535)][int]$Port8000 = 8000,
    [string]$Distro = 'QwenStation',
    [string]$ListenAddress
)

$ErrorActionPreference = 'Stop'
$FirewallName = 'Qwen38-vLLM-LAN-8000'
$FirewallMarker = 'Managed by setup-qwen-lan.ps1; Qwen38 vLLM LAN TCP 8000'
$StatePath = Join-Path (Join-Path $PSScriptRoot 'logs') 'setup-qwen-lan.state.json'
function Test-PrivateIPv4([string]$Address) {
    try { $bytes = ([System.Net.IPAddress]::Parse($Address)).GetAddressBytes() } catch { return $false }
    if ($bytes.Count -ne 4) { return $false }
    return ($bytes[0] -eq 10) -or
        ($bytes[0] -eq 172 -and $bytes[1] -ge 16 -and $bytes[1] -le 31) -or
        ($bytes[0] -eq 192 -and $bytes[1] -eq 168)
}

function Get-LanEndpoint([string]$Override) {
    if ($Override) {
        if (-not (Test-PrivateIPv4 $Override)) { throw "ListenAddress must be an RFC1918 IPv4 address." }
        $ip = @(Get-NetIPAddress -AddressFamily IPv4 -IPAddress $Override -ErrorAction SilentlyContinue)[0]
    } else {
        $route = @(Get-NetRoute -AddressFamily IPv4 -DestinationPrefix '0.0.0.0/0' |
            Where-Object { $_.NextHop -and $_.NextHop -ne '0.0.0.0' } |
            Sort-Object RouteMetric, InterfaceMetric)[0]
        if (-not $route) { throw 'No default IPv4 route was found.' }
        $ip = @(Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $route.InterfaceIndex |
            Where-Object { Test-PrivateIPv4 $_.IPAddress } | Select-Object -First 1)[0]
    }
    if (-not $ip) { throw 'No matching private IPv4 address was found.' }
    $profile = Get-NetConnectionProfile -InterfaceIndex $ip.InterfaceIndex -ErrorAction Stop
    if ($profile.NetworkCategory -ne 'Private') { throw "Interface $($ip.InterfaceIndex) is not a Private network profile." }
    $adapter = Get-NetAdapter -InterfaceIndex $ip.InterfaceIndex -ErrorAction Stop
    if ($adapter.Status -ne 'Up') { throw "Interface $($adapter.Name) is not up." }
    [pscustomobject]@{ Address = $ip.IPAddress; InterfaceAlias = $adapter.Name; InterfaceIndex = $ip.InterfaceIndex }
}

function Get-WslIPv4([string]$Name) {
    $raw = & wsl.exe -d $Name -- hostname -I 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Unable to query WSL distribution $Name." }
    $ips = ($raw -join ' ') -split '\s+' | Where-Object { $_ -and (Test-PrivateIPv4 $_) }
    $nat = @($ips | Where-Object { $_ -match '^172\.(1[6-9]|2\d|3[01])\.' })[0]
    if (-not $nat) { $nat = @($ips)[0] }
    if (-not $nat) { throw "No private WSL IPv4 address was found for $Name." }
    return $nat
}

function Invoke-PortProxy([string[]]$Arguments) {
    $output = & netsh @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "netsh portproxy failed: $($output -join ' ')" }
    return @($output)
}

function Get-PortProxyMappings {
    foreach ($line in (Invoke-PortProxy @('interface', 'portproxy', 'show', 'v4tov4'))) {
        if ($line -match '^\s*(?<ListenAddress>\S+)\s+(?<ListenPort>\d+)\s+(?<ConnectAddress>\S+)\s+(?<ConnectPort>\d+)\s*$') {
            [pscustomobject]@{ ListenAddress = $Matches.ListenAddress; ListenPort = [int]$Matches.ListenPort; ConnectAddress = $Matches.ConnectAddress; ConnectPort = [int]$Matches.ConnectPort }
        }
    }
}

function Test-FirewallRule($Rule, $Lan) {
    try {
        $port = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $Rule
        $address = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $Rule
        $interface = Get-NetFirewallInterfaceFilter -AssociatedNetFirewallRule $Rule
        return $Rule.Description -eq $FirewallMarker -and $Rule.Direction -eq 'Inbound' -and
            $Rule.Action -eq 'Allow' -and (@($Rule.Profile) -contains 'Private') -and
            (@($port.Protocol) -contains 'TCP') -and (@($port.LocalPort) -contains "$Port8000") -and
            (@($address.LocalAddress) -contains $Lan.Address) -and (@($address.RemoteAddress) -contains 'LocalSubnet') -and
            (@($interface.InterfaceAlias) -contains $Lan.InterfaceAlias)
    } catch { return $false }
}

function Ensure-FirewallRule($Lan) {
    $rules = @(Get-NetFirewallRule -DisplayName $FirewallName -ErrorAction SilentlyContinue)
    if ($rules.Count -gt 1) { throw "Multiple firewall rules use $FirewallName; refusing to choose one." }
    if ($rules.Count -eq 1) {
        if (-not (Test-FirewallRule $rules[0] $Lan)) { throw "Firewall rule $FirewallName exists but is not the managed rule." }
        return [pscustomobject]@{ Rule = $rules[0]; Created = $false }
    }
    $rule = New-NetFirewallRule -Name $FirewallName -DisplayName $FirewallName -Description $FirewallMarker -Direction Inbound -Action Allow -Enabled True -Protocol TCP -LocalPort $Port8000 -LocalAddress $Lan.Address -RemoteAddress LocalSubnet -InterfaceAlias $Lan.InterfaceAlias -Profile Private
    return [pscustomobject]@{ Rule = $rule; Created = $true }
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { return $null }
    try { $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json } catch { throw "Invalid state file $StatePath." }
    foreach ($name in @('ListenAddress', 'InterfaceAlias', 'WslAddress', 'Distro', 'FirewallName', 'FirewallRuleName')) {
        if (-not $state.$name) { throw "State file is missing $name." }
    }
    if ($state.Version -ne 1 -or $state.Script -ne 'setup-qwen-lan.ps1' -or $state.ListenPort -ne $Port8000 -or
        $state.FirewallName -ne $FirewallName -or $state.FirewallMarker -ne $FirewallMarker -or
        -not (Test-PrivateIPv4 $state.ListenAddress) -or -not (Test-PrivateIPv4 $state.WslAddress)) { throw 'State file failed validation.' }
    return $state
}

function Save-State($Lan, [string]$WslAddress, $Rule) {
    $directory = Split-Path -Parent $StatePath
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    [pscustomobject]@{
        Version = 1; Script = 'setup-qwen-lan.ps1'; ListenAddress = $Lan.Address; InterfaceAlias = $Lan.InterfaceAlias
        ListenPort = $Port8000; WslAddress = $WslAddress; Distro = $Distro; FirewallName = $FirewallName
        FirewallRuleName = $Rule.Name; FirewallMarker = $FirewallMarker
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run setup/apply/remove from an elevated Administrator PowerShell.' }
}

try {
    if ($Check -and $Remove) { throw 'Choose only one of -Check or -Remove.' }
    if (-not $Check) { Assert-Administrator }
    if ($Remove) {
        $state = Read-State
        if (-not $state) { throw "No managed state exists at $StatePath." }
        $mapping = @(Get-PortProxyMappings | Where-Object { $_.ListenAddress -eq $state.ListenAddress -and $_.ListenPort -eq $state.ListenPort -and $_.ConnectAddress -eq $state.WslAddress -and $_.ConnectPort -eq $state.ListenPort })
        if ($mapping.Count -ne 1) { throw 'Managed portproxy mapping is missing or changed; refusing removal.' }
        $lan = [pscustomobject]@{ Address = $state.ListenAddress; InterfaceAlias = $state.InterfaceAlias }
        $rule = Get-NetFirewallRule -Name $state.FirewallRuleName -ErrorAction SilentlyContinue
        if (-not $rule -or -not (Test-FirewallRule $rule $lan)) { throw 'Managed firewall rule is missing or changed; refusing removal.' }
        Invoke-PortProxy @('interface', 'portproxy', 'delete', 'v4tov4', "listenaddress=$($state.ListenAddress)", "listenport=$($state.ListenPort)") | Out-Null
        Remove-NetFirewallRule -Name $state.FirewallRuleName
        Remove-Item -LiteralPath $StatePath -Force
        Write-Host "Removed managed Qwen38 LAN endpoint on $($state.ListenAddress):$($state.ListenPort)."
        exit 0
    }

    $lan = Get-LanEndpoint $ListenAddress
    $wsl = Get-WslIPv4 $Distro
    $state = Read-State
    if ($Check) {
        $maps = @(Get-PortProxyMappings | Where-Object { $_.ListenAddress -eq $lan.Address -and $_.ListenPort -eq $Port8000 })
        $rules = @(Get-NetFirewallRule -DisplayName $FirewallName -ErrorAction SilentlyContinue)
        $service = Get-Service -Name iphlpsvc -ErrorAction Stop
        Write-Host "LAN: $($lan.InterfaceAlias) $($lan.Address) profile=Private"
        Write-Host "WSL: $Distro $wsl"
        Write-Host "iphlpsvc: $($service.Status) start=$($service.StartType)"
        Write-Host "portproxy matches: $($maps.Count); firewall rules named $($FirewallName): $($rules.Count)"
        if ($state) { Write-Host "managed state: $StatePath" } else { Write-Host 'managed state: absent' }
        $ready = $state -and $maps.Count -eq 1 -and $rules.Count -eq 1 -and $service.Status -eq 'Running'
        if ($ready) {
            $ready = $maps[0].ConnectAddress -eq $wsl -and $maps[0].ConnectPort -eq $Port8000 -and
                $state.ListenAddress -eq $lan.Address -and $state.WslAddress -eq $wsl -and
                $rules[0].Enabled -eq 'True' -and (Test-FirewallRule $rules[0] $lan)
        }
        if ($ready) { exit 0 } else { exit 2 }
    }

    $existing = @(Get-PortProxyMappings | Where-Object { $_.ListenAddress -eq $lan.Address -and $_.ListenPort -eq $Port8000 })
    if ($existing.Count -gt 1) { throw 'Multiple mappings use the requested listen address and port.' }
    if ($existing.Count -eq 1 -and -not $state) { throw 'An existing portproxy mapping has no managed ownership record; nothing changed.' }
    $oldMapping = $null; $mappingRemoved = $false; $mappingAdded = $false; $createdFirewall = $null
    try {
        if ($existing.Count -eq 1 -and ($existing[0].ConnectAddress -ne $wsl -or $existing[0].ConnectPort -ne $Port8000)) {
            if (-not $state -or $state.ListenAddress -ne $lan.Address -or $state.InterfaceAlias -ne $lan.InterfaceAlias -or $existing[0].ConnectAddress -ne $state.WslAddress) { throw 'An unowned portproxy mapping conflicts with the requested endpoint.' }
            $oldMapping = $existing[0]
            Invoke-PortProxy @('interface', 'portproxy', 'delete', 'v4tov4', "listenaddress=$($lan.Address)", "listenport=$Port8000") | Out-Null
            $existing = @(); $mappingRemoved = $true
        }
        $service = Get-Service -Name iphlpsvc -ErrorAction Stop
        if ($service.StartType -eq 'Disabled') { Set-Service -Name iphlpsvc -StartupType Manual }
        if ($service.Status -ne 'Running') { Start-Service -Name iphlpsvc }
        if ($existing.Count -eq 0) {
            Invoke-PortProxy @('interface', 'portproxy', 'add', 'v4tov4', "listenaddress=$($lan.Address)", "listenport=$Port8000", "connectaddress=$wsl", "connectport=$Port8000") | Out-Null
            $mappingAdded = $true
        }
        $firewall = Ensure-FirewallRule $lan
        if ($firewall.Created) { $createdFirewall = $firewall.Rule }
        Save-State $lan $wsl $firewall.Rule
    } catch {
        if ($createdFirewall) { try { Remove-NetFirewallRule -Name $createdFirewall.Name -ErrorAction SilentlyContinue } catch {} }
        if ($mappingAdded) { try { Invoke-PortProxy @('interface', 'portproxy', 'delete', 'v4tov4', "listenaddress=$($lan.Address)", "listenport=$Port8000") | Out-Null } catch {} }
        if ($mappingRemoved -and $oldMapping) { try { Invoke-PortProxy @('interface', 'portproxy', 'add', 'v4tov4', "listenaddress=$($oldMapping.ListenAddress)", "listenport=$($oldMapping.ListenPort)", "connectaddress=$($oldMapping.ConnectAddress)", "connectport=$($oldMapping.ConnectPort)") | Out-Null } catch {} }
        throw
    }
    Write-Host "Qwen38 vLLM LAN endpoint: http://$($lan.Address):$Port8000 -> WSL $($wsl):$Port8000"
    Write-Host "Firewall: $FirewallName (Private profile, LocalSubnet only)"
} catch {
    Write-Error $_
    exit 1
}

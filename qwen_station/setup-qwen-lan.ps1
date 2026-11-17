# Opens the Windows firewall for the API of Qwen Station, so that other devices can reach it.
#
#   -ListenAddress 0.0.0.0        every address of this PC (private and domain networks only)
#   -ListenAddress 192.168.1.20   one address of this PC (whatever Windows calls its network)
#   -Check                        exit code 0 if the rule for the given address and port is in place, else 2
#   -Remove                       take the rule back
#
# The model runs inside WSL, which Windows forwards to 127.0.0.1 only. Qwen Station itself listens on the chosen
# address and passes connections on, so nothing else is needed: no port forwarding, no service.
#
# The script changes only what it created itself: its firewall rule carries a marker. Versions before 1.4.0 also
# created a `netsh portproxy` forwarding to the WSL network; the one that belongs to the marked rule is removed.
param(
    [switch]$Check,
    [switch]$Remove,
    [ValidateRange(1, 65535)][int]$Port8000 = 8000,
    [string]$ListenAddress = '0.0.0.0'
)

$ErrorActionPreference = 'Stop'
$FirewallName = 'Qwen38-vLLM-LAN-8000'
$FirewallMarker = 'Managed by setup-qwen-lan.ps1; Qwen38 vLLM LAN TCP 8000'
$StatePath = Join-Path (Join-Path $PSScriptRoot 'logs') 'setup-qwen-lan.state.json'
$AnyAddress = '0.0.0.0'

function Test-IPv4([string]$Address) {
    $parsed = $null
    return $Address -match '^\d+\.\d+\.\d+\.\d+$' -and [System.Net.IPAddress]::TryParse($Address, [ref]$parsed)
}

function Assert-ListenAddress([string]$Address) {
    if (-not (Test-IPv4 $Address)) { throw "ListenAddress must be an IPv4 address, got '$Address'." }
    if ($Address -eq $AnyAddress) { return }
    if ($Address -like '127.*') { throw 'A loopback address needs no firewall rule.' }
    $own = @(Get-NetIPAddress -AddressFamily IPv4 -IPAddress $Address -ErrorAction SilentlyContinue)
    if ($own.Count -eq 0) { throw "This PC has no address $Address." }
}

# The managed firewall rule, or $null. A rule of that name without the marker is someone else's: stop.
function Get-ManagedRule {
    $rule = Get-NetFirewallRule -Name $FirewallName -ErrorAction SilentlyContinue
    if ($rule -and $rule.Description -ne $FirewallMarker) { throw "A firewall rule named $FirewallName exists but was not created by Qwen Station; nothing changed." }
    return $rule
}

# The local address the rule is for, as a listen address: 0.0.0.0 for "Any".
function Get-RuleAddress($Rule) {
    $local = @((Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $Rule).LocalAddress)[0]
    if (-not $local -or $local -eq 'Any') { return $AnyAddress }
    return "$local"
}

function Get-RulePort($Rule) {
    return "$(@((Get-NetFirewallPortFilter -AssociatedNetFirewallRule $Rule).LocalPort)[0])"
}

function Test-ManagedRule($Rule, [string]$Address) {
    if (-not $Rule -or "$($Rule.Enabled)" -ne 'True' -or "$($Rule.Direction)" -ne 'Inbound' -or "$($Rule.Action)" -ne 'Allow') { return $false }
    $filter = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $Rule
    if (-not (@($filter.Protocol) -contains 'TCP') -or (Get-RulePort $Rule) -ne "$Port8000") { return $false }
    if ((Get-RuleAddress $Rule) -ne $Address) { return $false }
    if (@((Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $Rule).RemoteAddress)[0] -ne 'Any') { return $false }
    # Profile is a set of flags: Domain 1, Private 2, Public 4, and 0 for all of them.
    $wanted = if ($Address -eq $AnyAddress) { 3 } else { 0 }
    return ([int]$Rule.Profile) -eq $wanted
}

function New-ManagedRule([string]$Address) {
    $common = @{ Name = $FirewallName; DisplayName = $FirewallName; Description = $FirewallMarker; Direction = 'Inbound'
                 Action = 'Allow'; Enabled = 'True'; Protocol = 'TCP'; LocalPort = $Port8000 }
    if ($Address -eq $AnyAddress) {
        # Every address of this PC, but not on a network Windows calls public (a hotel, a cafe).
        New-NetFirewallRule @common -Profile Domain, Private | Out-Null
    } else {
        New-NetFirewallRule @common -LocalAddress $Address -Profile Any | Out-Null
    }
}

function Get-PortProxyMappings {
    $output = & netsh interface portproxy show v4tov4 2>&1
    if ($LASTEXITCODE -ne 0) { return }
    foreach ($line in $output) {
        if ($line -match '^\s*(?<ListenAddress>\S+)\s+(?<ListenPort>\d+)\s+(?<ConnectAddress>\S+)\s+(?<ConnectPort>\d+)\s*$') {
            [pscustomobject]@{ ListenAddress = $Matches.ListenAddress; ListenPort = $Matches.ListenPort; ConnectAddress = $Matches.ConnectAddress; ConnectPort = $Matches.ConnectPort }
        }
    }
}

# Forwardings of versions before 1.4.0: on the address and port of the marked rule, to the same port in WSL.
# Such a forwarding holds the address, so Qwen Station could not listen there.
function Get-OldMappings($Rule) {
    if (-not $Rule) { return @() }
    $address = Get-RuleAddress $Rule
    $port = Get-RulePort $Rule
    return @(Get-PortProxyMappings | Where-Object { $_.ListenAddress -eq $address -and $_.ListenPort -eq $port -and $_.ConnectPort -eq $port })
}

function Remove-OldMappings($Rule) {
    foreach ($mapping in (Get-OldMappings $Rule)) {
        & netsh interface portproxy delete v4tov4 "listenaddress=$($mapping.ListenAddress)" "listenport=$($mapping.ListenPort)" | Out-Null
        Write-Host "Removed the old port forwarding $($mapping.ListenAddress):$($mapping.ListenPort) -> $($mapping.ConnectAddress):$($mapping.ConnectPort)."
    }
}

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run this from an elevated Administrator PowerShell.' }
}

try {
    if ($Check -and $Remove) { throw 'Choose only one of -Check or -Remove.' }
    if (-not $Check) { Assert-Administrator }
    $rule = Get-ManagedRule

    if ($Remove) {
        Remove-OldMappings $rule
        if ($rule) { Remove-NetFirewallRule -Name $FirewallName }
        if (Test-Path -LiteralPath $StatePath) { Remove-Item -LiteralPath $StatePath -Force }
        Write-Host 'Removed the firewall rule of Qwen Station.'
        exit 0
    }

    Assert-ListenAddress $ListenAddress
    if ($Check) {
        $ruleReady = Test-ManagedRule $rule $ListenAddress
        $old = @(Get-OldMappings $rule).Count
        Write-Host "Firewall rule for $($ListenAddress):$($Port8000): $ruleReady; old port forwardings: $old"
        if ($ruleReady -and $old -eq 0 -and (Test-Path -LiteralPath $StatePath)) { exit 0 } else { exit 2 }
    }

    Remove-OldMappings $rule
    if ($rule) { Remove-NetFirewallRule -Name $FirewallName }
    New-ManagedRule $ListenAddress
    # The app looks for this file to know that there is something to take back later.
    New-Item -ItemType Directory -Path (Split-Path -Parent $StatePath) -Force | Out-Null
    [pscustomobject]@{ Version = 3; Script = 'setup-qwen-lan.ps1'; ListenAddress = $ListenAddress; ListenPort = $Port8000; FirewallName = $FirewallName } |
        ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    Write-Host "The firewall allows TCP $($ListenAddress):$Port8000 for Qwen Station."
} catch {
    Write-Error $_
    exit 1
}

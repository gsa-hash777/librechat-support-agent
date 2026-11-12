param([string]$Root = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))), [switch]$Trust)
$ErrorActionPreference = 'Stop'
$directory = Join-Path $Root '.local/support-tls'
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$certPath = Join-Path $directory 'server.crt'
$keyPath = Join-Path $directory 'server.key'
if ((Test-Path -LiteralPath $certPath) -ne (Test-Path -LiteralPath $keyPath)) { throw 'Incomplete local TLS pair; refusing to overwrite.' }
if (-not (Test-Path -LiteralPath $certPath)) {
    $rsa = [System.Security.Cryptography.RSA]::Create(3072)
    $req = [System.Security.Cryptography.X509Certificates.CertificateRequest]::new('CN=LibreChat Support Local', $rsa, [System.Security.Cryptography.HashAlgorithmName]::SHA256, [System.Security.Cryptography.RSASignaturePadding]::Pkcs1)
    $san = [System.Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
    $san.AddDnsName('localhost'); $san.AddDnsName('support-tls'); $san.AddIpAddress([System.Net.IPAddress]::Loopback)
    $req.CertificateExtensions.Add($san.Build())
    $req.CertificateExtensions.Add([System.Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($false, $false, 0, $true))
    $req.CertificateExtensions.Add([System.Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new([System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature, $true))
    $eku = [System.Security.Cryptography.OidCollection]::new()
    $eku.Add([System.Security.Cryptography.Oid]::new('1.3.6.1.5.5.7.3.1')) | Out-Null
    $req.CertificateExtensions.Add([System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new($eku, $true))
    $cert = $req.CreateSelfSigned([DateTimeOffset]::UtcNow.AddMinutes(-5), [DateTimeOffset]::UtcNow.AddDays(180))
    [System.IO.File]::WriteAllText($certPath, $cert.ExportCertificatePem())
    [System.IO.File]::WriteAllText($keyPath, $rsa.ExportPkcs8PrivateKeyPem())
    $rsa.Dispose()
}
$cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::CreateFromPem([System.IO.File]::ReadAllText($certPath))
if ($cert.NotAfter.ToUniversalTime() -le [DateTime]::UtcNow) { throw 'Local TLS certificate expired; rotate the pair deliberately.' }
if ($Trust) {
    # Trust only this non-CA server certificate, only for the current Windows user.
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new('Root', 'CurrentUser')
    $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
    try { $store.Add($cert) } finally { $store.Close() }
}
@{ thumbprint=$cert.Thumbprint; expires_utc=$cert.NotAfter.ToUniversalTime().ToString('o'); current_user_trusted=[bool]$Trust; certificate=$certPath } | ConvertTo-Json

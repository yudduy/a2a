# Security Policy

We take security seriously and appreciate your efforts to responsibly disclose vulnerabilities.

## Supported Versions
We provide security updates for the latest minor release series.

## Reporting a Vulnerability
- Please report vulnerabilities via GitHub Security Advisories or email `security@your-org.com`.
- Include a detailed description, reproduction steps, impacted versions, and any PoC if available.
- We will acknowledge receipt within 2 business days.

## Disclosure and Response Timeline
- Triage within 5 business days.
- Fix development timeline communicated within 10 business days.
- Coordinated disclosure date agreed with reporter; credits provided where appropriate.

## Security Best Practices
- Use a managed secrets store; never commit secrets to the repository.
- Enforce least-privilege API keys with rotation and environment-specific provisioning.
- Require TLS for all endpoints; enable JWT verification with strict algorithms and claims.
- Apply input validation and output redaction to mitigate prompt injection and data leakage.

## PGP / Encryption
If you require encrypted communication, please request our PGP key via `security@your-org.com`.



# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.1.x   | :white_check_mark: |
| < 1.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in ARIA, please report it responsibly.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead:

1. Email the maintainers with a description of the vulnerability, or
2. Use GitHub's **private vulnerability reporting** feature (Security tab -> "Report a vulnerability").

### What to include

- Type of vulnerability (e.g., unsafe deserialization, path traversal, dependency CVE)
- Steps to reproduce or a proof of concept
- Potential impact assessment

### Response timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Resolution target**: within 30 days for critical issues

## Security Considerations for Drone Autopilot Software

ARIA is a research/educational codebase. Before using any autonomous flight
software on real hardware:

- Conduct a full threat model (RF jamming, GPS spoofing, sensor spoofing)
- Implement independent hardware failsafes (geofence, RTH, kill switch)
- Never fly autonomous missions beyond visual line of sight without
  appropriate regulatory approval
- Treat LLM-generated flight plans as **advisory input only**, always
  validated by deterministic guardrails (Agent C) before execution

# Security policy

## Supported versions

Security fixes are applied to the latest release on the default branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this
repository. Do not include credentials, private media, private prompts,
internal hostnames, or production logs in a public issue.

## Deployment guidance

VidBangerGen is designed for trusted local or private-network deployment.

- Do not expose an unauthenticated ComfyUI endpoint to the public internet.
- Keep `.env`, runtime databases, uploads, outputs, model inventories, and logs
  outside version control.
- Put an authenticated TLS reverse proxy in front of VidBangerGen when remote
  browser access is required.
- Use least-privilege service accounts and separate writable directories for
  ComfyUI input/output data.
- Treat uploaded media and generated artifacts as private user data.
- Review optional installation scripts before allowing them to modify a
  ComfyUI environment.

The application rejects credentials embedded in ComfyUI URLs. SSH integration
uses non-interactive key-based access and is optional.

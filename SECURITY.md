# Security policy

## Supported versions

Fixes land on `main`. There are no maintained release branches.

## Reporting a vulnerability

Please report privately, not in a public issue.

Use **Security → Report a vulnerability** on this repository. That opens a private
advisory visible only to the maintainer. If that option is not shown, open an issue that
says only that you have a security report and asks for a private channel. Leave the
details out of it.

Include what you found, how to reproduce it, and what an attacker could do with it.

## What is in scope

This project runs on your own machine. The parts worth a security report are the ones
that accept input from somewhere else:

- **`_serve.py`, the local server.** It listens on all network interfaces (HTTP on 8137,
  HTTPS on 8138) so that a phone on the same Wi-Fi can reach the capture page. It is a
  development server for a network you trust. Reports of ways to write files, start runs
  or read outside the repository from another host are in scope.
- **The Ground Control Station's backend proxy** (`groundcontrol/app/api/backend`), which
  only forwards an allow-list of routes and refuses writes that do not originate from the
  local application.
- **File parsers** that read uploaded or imported data: videos, pose logs, telemetry,
  point clouds, meshes and design surfaces.

## What is not a vulnerability here

- The HTTPS certificate is self-signed. Phone browsers require HTTPS for camera and motion
  access, and there is no way to issue a trusted certificate for a LAN address.
- The server has no authentication. Do not expose ports 8137 or 8138 to the internet, and
  do not run it on a network you do not control.

## Vendored binaries

`tools/` contains third-party builds of COLMAP and FFmpeg. Vulnerabilities in those belong
upstream, but an issue here asking for a version bump is welcome.

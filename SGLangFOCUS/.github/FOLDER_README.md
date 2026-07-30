# GitHub Maintenance Files

This folder keeps lightweight GitHub metadata for the FOCUS fork.

The upstream SGLang GitHub Actions workflow set has been removed from this
branch. Those workflows rely on upstream-only runners, secrets, release
credentials, and CI permission automation, so they are not useful checks for the
open-source FOCUS artifact. See `workflows/README.md`.

## CI Permissions

`CI_PERMISSIONS.json` and the related scripts are inherited from upstream
SGLang. They are kept for reference only unless a maintainer reintroduces a
compatible CI setup.

## Others
- `MAINTAINER.md` defines the code maintenance model.

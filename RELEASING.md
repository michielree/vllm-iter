# Releasing

`pyproject.toml` is the source of truth for the package version. Releases are
published from Git tags named `vX.Y.Z`, and the release workflow rejects tags
that do not match the package version.

## Version policy

- Patch releases, such as `0.1.0` to `0.1.1`, are for bug fixes and
  packaging-only fixes.
- Minor releases, such as `0.1.0` to `0.2.0`, are for new features or visible
  behavior changes.
- Major releases are reserved for breaking changes after `1.0.0`.
- Stay in `0.x` until `IterableLLM.generate_iter()` is stable across real vLLM
  versions.

## Release steps

1. Update `version` in `pyproject.toml`.
2. Move completed entries from `CHANGELOG.md`'s `Unreleased` section into a new
   `X.Y.Z` section.
3. Run the local checks:

   ```bash
   uv run pytest -m "not slow and not integration"
   uv build
   uv run --with twine twine check dist/*
   ```

4. Commit the release:

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release X.Y.Z"
   ```

5. Tag and push the release:

   ```bash
   git tag vX.Y.Z
   git push origin main
   git push origin vX.Y.Z
   ```

GitHub Actions publishes the package to PyPI from the tag using Trusted
Publishing. Configure the PyPI publisher with repository `michielree/vllm-iter`,
workflow `release.yml`, and environment `pypi`.

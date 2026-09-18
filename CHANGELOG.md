# Changelog

All notable changes to ModelDirector are recorded here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/) — once we cut 1.0.

## [Unreleased]

### Fixed
- README clone URL: `aniketkarne-com/ModelDirector` -> `aniketkarne/ModelDirector`
  (line 102 and the Issues link at line 446). The old URL 404'd for every
  new user running the quickstart.
- README test-count bullet: was 43 unit tests, actually 47.
- README same-vendor selector recommendation: `claude-3.5-haiku` scoring
  an Anthropic-heavy candidate set has a known same-vendor scoring bias.
  Both `examples/config.yaml` and the README now ship with
  `openai/gpt-4o-mini` as the cross-vendor selector by default.

### Added
- `.github/workflows/tests.yml`: runs `pytest tests/ -m "not integration"`
  on Python 3.11 and 3.12 for every push and PR. The README's
  "CI on every commit runs the unit suite" claim is now backed by a
  workflow instead of wishful thinking.

## [0.1.0] - 2026-06-07

Initial release.
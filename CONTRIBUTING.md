# Contributing to Mira

Use [Issues](https://github.com/Xopoko/Mira/issues) for bugs and feature ideas.
For a bug, include the steps, the expected and observed behavior, and the
relevant Windows, Python or iOS version. Use the
[private reporting instructions](SECURITY.md) for security issues.

Start with [the source setup](README.md#build-from-source) and the
[architecture guide](docs/architecture.md). The
[development guide](docs/development.md) lists the test commands and describes
where Mira writes local data. Changes to the iPhone companion also need the
[iOS checks](phone/README.md#checks).

Keep a pull request focused on one change. Explain what someone using Mira
will notice and how you checked it. Add a focused regression test when
existing coverage misses a behavior change. Use synthetic data in tests and
review logs and screenshots for private details before attaching them.

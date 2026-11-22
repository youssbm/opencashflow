# Security policy

## Supported versions

Security fixes are applied to the latest `0.1.x` release. This project is pre-1.0 and APIs may change.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately to the maintainers
through the repository's Security tab ("Report a vulnerability").

Include:

* affected version or commit;
* a minimal reproduction;
* impact assessment;
* any suggested fix.

We aim to acknowledge reports within 72 hours and to publish a coordinated fix.

## Scope

Relevant issues include:

* arbitrary code execution through dataset loaders, archives or model files;
* unsafe deserialisation through joblib/pickle artifacts;
* credential or personal-data exposure;
* supply-chain risks in optional dependencies;
* path traversal in archive extraction.

## Data and model handling

* Do not commit raw bank data, credentials, checkpoints or personal data.
* Downloaded datasets are written to `data/raw` and are gitignored.
* `CashFlowPipeline.save` uses joblib, which can execute code on load. Only load artifacts you
  trust. This is documented in the pipeline docstring.
* LimiX weights are loaded from a local checkpoint. Verify the source and commit before use.
* The LimiX checkpoint is not bundled or downloaded automatically because of its licence and size.

## Dependency hygiene

Optional dependencies are imported lazily. If you add one, pin a minimum version, document why it
is needed and check its licence. The project does not vendor third-party source code.

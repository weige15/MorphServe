# Alignment guard attempt 1

The new misaligned-region safety test passed and the normal in-place copy/restore test passed. The pre-existing tiny-tail test failed only because the new earlier guard reports `No aligned space available...` rather than the older `No space available...`; both are correct fail-closed outcomes.

Retry change: broaden that assertion to accept either explicit no-space diagnostic. No implementation or threshold change.

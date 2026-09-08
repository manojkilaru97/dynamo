# Upstream provenance

Unmodified base: crates.io dynamo-parsers7.0.1, archive SHA256
97cce1ec70f4c4896ff9a4e0bb29cd603f465e599c94bf2085b71872a3a58c70.
VCS commit293e0222546d49179162c365b189b01dea00f700, path parsers/v1.

Local patch: JailedStream::apply emits incoming logprobs exactly once in
per-choice input order, independent of transformed content buffering. Six
regressions in the same source file cover complete metadata records, partial
markers, role/reasoning/logprob-only chunks, multi-call prefixes/suffixes,
malformed EOF, refusal metadata, and interleaved choices.

Patched src/tool_calling/jail/mod.rs SHA256:
deac18d4602c4a196915b86263821fe12306008e970ff9ccca9762be4ef834ef.

License and all unrelated upstream files are preserved. Registry checksum and
installation-state files are not used to claim integrity of modified source.

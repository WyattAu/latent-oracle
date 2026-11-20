# Pattern adoption

Declared per references/adoption-format.md; IDs resolve against
references/catalog.md in pattern-standards.

| Pattern | Status | Evidence |
|---|---|---|
| T49 | adopted | src/engine_oracle/engine.cpp:150 (symbolic gates verify the neural move choice; ARCHITECTURE.md:5) |
| P65 | adopted | tests/perft (perft-suite gated movegen; known node counts as golden masters, ARCHITECTURE.md:12) |
| P67 | adopted | tests/fuzz_movegen.py:1 (differential movegen fuzz over generated legal positions against python-chess) |
| P110 | adopted | ARCHITECTURE.md:29 (piece-code shard convention shared across engine, data pipeline, and trainer) |
| P111 | adopted | tests/ci_parity.py:1 (FP32 and INT8 parity gate against the trainer; ARCHITECTURE.md:27) |

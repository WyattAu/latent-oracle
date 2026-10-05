# Contributing

- cmake --preset debug && cmake --build --preset debug && ctest --preset debug
- Full gate: ./build/debug/latent-oracle perftsuite (release-grade movegen)
- Any nn/ layout change: run the parity harnesses (FP32 + INT8) before push
- Python research tooling lives in the latent-oracle-data repo
- Never edit armed pipeline scripts a running process may re-read

This is an implementation of [picoGPT](https://github.com/jaymody/picoGPT) in Rust.

### Dependencies
In addition to the crates in the `Cargo.toml` file, this project relies on the python
scripts in utils.py to download the GPT2 weights. You will therefore need all of the
imports from the `utils.py` file to be available in your python environment.

This project uses [pyo3](https://pyo3.rs/v0.23.5/) to bind the python interpreter to rust.

### How to Run

After downloading the repo, you can build and run the program with the following commands:
```
cargo run --  --prompt [*prompt*] --n-tokens-to-generate [*n_tokens*]
```

For example:
```
cargo run --  --prompt "Alan Turing theorized that computers would one day become" --n-tokens-to-generate 10
```

For faster performance, you can compile the program for release:
```
cargo build --release
target/release/rustygpt --prompt "Alan Turing theorized that computers would one day become" --n-tokens-to-generate 10
```
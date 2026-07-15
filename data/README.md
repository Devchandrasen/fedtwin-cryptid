# Local data layout

No media, embeddings, descriptors, private labels, or downloaded archives are committed to this repository.

The runners use these default local paths:

```text
data/
  vcsl_features/
    isc_extracted/       # locally extracted released ISC frame descriptors
  fma/
    fma_small/           # locally obtained FMA audio files
    fma_metadata/        # locally obtained FMA metadata
    cache/               # generated audio-feature cache

public_data/
  vcsl_metadata/         # VCSL pair files, category map, frame metadata
```

Acquire VCSL metadata/descriptors and FMA from their official project distributions, review their licenses, and stage only the paths required by the selected tier. The repository deliberately does not automate acceptance of third-party terms or redistribute the resulting files.

All paths are explicit and can be overridden:

```powershell
python run_benchmark.py --help
```

Validate staged inputs before a run:

```powershell
python -m fedtwin.cli reproduce --config configs\paper.yaml --data-root D:\datasets --output-dir outputs\paper-rerun
```

Each adapter checks the required schema, dimensions, missing files, supplied hashes, license note, and train/test identity policy. Follow the licenses and access conditions of VCSL, ISC descriptors, and FMA. Do not commit downloaded content or derived embeddings.

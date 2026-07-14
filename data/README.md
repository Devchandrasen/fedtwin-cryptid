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

All paths can be overridden:

```powershell
python run_benchmark.py --help
```

Follow the licenses and access conditions of VCSL, ISC descriptors, and FMA. Do not commit downloaded content or derived embeddings.

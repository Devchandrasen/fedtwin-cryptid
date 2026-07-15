# Licensed data acquisition

The repository never redistributes raw media or downloaded frame descriptors. Use a data root outside the Git checkout and review the upstream terms before downloading. The commands below pin the sources used by the paper protocol.

## VCSL metadata and labels

The public metadata are taken from the official `alipay/VCSL` repository at commit `29ce63909ce605a73335ce48d1e21f395b4b503c`. Disable automatic line-ending conversion so the byte-level hashes are identical on Windows and Linux.

```powershell
$DataRoot = "D:\fedtwin-data"
$Download = Join-Path $DataRoot "_downloads\VCSL"
git -c core.autocrlf=false clone https://github.com/alipay/VCSL.git $Download
git -C $Download checkout 29ce63909ce605a73335ce48d1e21f395b4b503c
$Target = Join-Path $DataRoot "public_data\vcsl_metadata"
New-Item -ItemType Directory -Force -Path $Target | Out-Null
@(
  "frames_all.csv",
  "pair_file_test.csv",
  "pair_file_train.csv",
  "pair_file_val.csv",
  "videos_url_uuid.csv",
  "video_categories.json"
) | ForEach-Object {
  Copy-Item -LiteralPath (Join-Path $Download "data\$_") -Destination (Join-Path $Target $_)
}
```

Expected SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `frames_all.csv` | `e9971d4bd08a31b03bc73fdf44b2f9d15e59ed7043b14af5ca410f2606e180b6` |
| `pair_file_test.csv` | `b52c1b5fb394d117d7f02206493331e78ba26fdb7ee344fc4302aba0cf231880` |
| `pair_file_train.csv` | `e7fb88859e29dc872c10fc2b9e1a1a5c5a6516a6346ec685aec42ecef14a5c9c` |
| `pair_file_val.csv` | `f7065467eb71340576c1f83ce6ada8810b02311d8d932c269069c414fa746f91` |
| `videos_url_uuid.csv` | `f424111d2b5d20979f92472af1f5544a8fcfb168682236169e205d284c11a869` |
| `video_categories.json` | `112043737dfde578d3950bd9c29ab272d0e473cfc73f7eb1c70433f12c10bb42` |

## VCSL ISC descriptors

The official VCSL repository lists the original ISC descriptor archive in `data/vcsl_features.txt`:

```text
https://drive.google.com/file/d/1OE3JqV_h6EkGfvaSv13C97hNI4-bql95/view
```

The archive used for the paper rerun has SHA-256
`069b973c3e809b2fe48cbdb5d8c348fb0151f8f9bc738fd328d354ce232944c7`.
Download and extract it under:

```text
<DATA_ROOT>/data/vcsl_features/isc_extracted/
```

For example, after downloading the file as `vcsl_isc_features.tar.gz`:

```powershell
$Archive = Join-Path $DataRoot "_downloads\vcsl_isc_features.tar.gz"
$Target = Join-Path $DataRoot "data\vcsl_features\isc_extracted"
Get-FileHash -Algorithm SHA256 -LiteralPath $Archive
New-Item -ItemType Directory -Force -Path $Target | Out-Null
tar -xzf $Archive -C $Target
```

The adapter recursively locates the `eff256d` directory, validates every NumPy array without pickle support, checks a single finite descriptor dimension across the complete archive, and records a SHA-256 hash for every descriptor file used by the run.

## FMA-small audio

FMA metadata are released under CC BY 4.0. The audio remains under each artist's selected license and is for research use. The official project provides these archives and SHA-1 checksums:

```powershell
$DataRoot = "D:\fedtwin-data"
$Download = Join-Path $DataRoot "_downloads"
New-Item -ItemType Directory -Force -Path $Download | Out-Null
$MetadataZip = Join-Path $Download "fma_metadata.zip"
$SmallZip = Join-Path $Download "fma_small.zip"
curl.exe -L --fail --retry 5 -o $MetadataZip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl.exe -L --fail --retry 5 -o $SmallZip https://os.unil.cloud.switch.ch/fma/fma_small.zip
Get-FileHash -Algorithm SHA1 -LiteralPath $MetadataZip
Get-FileHash -Algorithm SHA1 -LiteralPath $SmallZip
```

Expected SHA-1 values:

- `fma_metadata.zip`: `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`
- `fma_small.zip`: `ade154f733639d52e35e32f5593efe5be76c6d70`

Extract to:

```text
<DATA_ROOT>/data/fma/fma_metadata/
<DATA_ROOT>/data/fma/fma_small/
```

### Pinned mirror route used by the confirmatory rerun

The official audio host may be unavailable. The confirmatory rerun therefore uses the
FMA-small-compatible redistribution pack
benjamin-paine/free-music-archive-small at immutable revision
d291bdb4c842bd5f007c60f9d3d6ba73199cd1c0. The pack maintainer reports that six
unreadable tracks and 78 tracks with unclear redistribution terms were removed. This
route is disclosed as a curated mirror, not as a byte-identical copy of
fma_small.zip.

Download the first three configured Parquet shards into a directory outside the Git
checkout. Their exact sizes and SHA-256 hashes are frozen in
configs/fma_hf_extraction.json. The following command verifies each shard before
extracting 1,584 MP3 payloads and records a hash for every output:

~~~powershell
uv run --with pyarrow==25.0.0 python tools\extract_hf_fma_small.py --config configs\fma_hf_extraction.json --shard-dir (Join-Path $DataRoot "_downloads\hf_fma_small") --output-dir (Join-Path $DataRoot "data\fma\fma_small") --manifest (Join-Path $DataRoot "data\fma\fma_small_extraction_manifest.json")
~~~

The paper configuration deterministically selects 1,200 tracks in sorted round-robin
order over the official top-level genre field. The adapter checks that every staged
track is in the official small subset, has a top-level genre and non-empty
per-track license, rejects duplicate identifiers, and hashes all staged audio.

The first experiment run creates a non-executable, schema-validated descriptor cache
and sidecar provenance manifest. Cache reuse revalidates every selected source input.
Decode failures are written to a deterministic JSON ledger, and the paper
configuration fails if more than 5% of selected tracks cannot be decoded.

## Reproduction

```powershell
python -m fedtwin.cli reproduce `
  --config configs\paper.yaml `
  --data-root $DataRoot `
  --output-dir outputs\paper-rerun
```

Do not copy the download directory, extracted media, descriptor arrays, or generated FMA cache into `paper_results/`, a Git commit, an Overleaf archive, or a release ZIP.

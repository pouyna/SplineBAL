# Result caches

Every script reads and writes its `results_*.json` cache in this folder (`controls.py band`
also writes `results_band_raw.npz`). The caches are about 40 MB in total and are not kept in
the repository itself.

* To **download** them, take `SplineBAL-caches-v1.0.zip` from the repository's
  [v1.0.0 release](https://github.com/pouyna/SplineBAL/releases/tag/v1.0.0) and unzip it
  in the repository root, so that the files land in this folder.

* To **regenerate** them, run the pipeline from the `code` folder (about a day at
  `--jobs=3` on a 16 GB machine):

  ```bash
  python3 reproduce.py campaigns --jobs=3
  ```

  Every run is independently seeded, so the result does not depend on the worker count.

* To **check a single result** without the full pipeline, run the campaign subcommand
  listed in the top-level README; it writes its cache here, and
  `recompute_stats.py <section>` prints the numbers derived from it.

# Per-machine measured results

Every machine-dependent number and data-driven figure in the deck comes from
`results/<MACHINE>/`, selected by `scripts/deck_config.py` (`DEFAULT_MACHINE`,
or the `DECK_MACHINE` env var).

- `RTX_A6000/` - archive of the development results. NOTE: the actual device is
  the **RTX 6000 Ada** (AD102, CC 8.9), not the Ampere "RTX A6000"; the folder
  name is just a label. See `RTX_A6000/device.txt`.
- `B200/` - the results the **paper references**. Populate with
  `scripts/collect_results.py`; see `B200/README.md`.

Each machine folder holds: `numbers.json` (scalars used in slide notes),
`dimuon_timeline.json` (nsys trace for the timeline figure),
`bench_ak_argmin.json` (argmin figure), `device.txt`.

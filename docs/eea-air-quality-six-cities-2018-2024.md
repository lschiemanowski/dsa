# EEA Air Quality — Six European Cities, 2018–2024

This is DSA's third versioned evaluation pack. Version `1.1.0` combines a content-pinned
DuckDB database with one unified set of 100 exact-JSON problems; there is no
core/generalization split.

## Data

The database contains hourly European Environment Agency (EEA) primary validated E1a
observations for Berlin, Madrid, Paris, Rome, Stockholm, and Warsaw. It covers the
half-open interval from `2018-01-01T00:00:00` through `2025-01-01T00:00:00` in the
EEA's fixed UTC+1 reporting time and includes SO2, PM10, O3, NO2, CO, and PM2.5.

The 315.3 MiB DuckDB database contains:

- 18,566,444 raw measurement rows from 18 source-request archives;
- 18,136,808 measurements accepted by the validity rule;
- 364 distinct sampling-point identifiers;
- 1,926,743 materialized city-hour rows;
- 80,417 daily, 2,644 monthly, and 222 annual aggregate rows; and
- source requests, archive sizes, checksums, row counts, and coverage in the
  `metadata` schema.

For ordinary analysis, use `analysis.city_hourly`, `analysis.city_daily`,
`analysis.city_monthly`, and `analysis.city_annual`. Sampling-point questions use
`analysis.valid_measurements`; validity audits use `raw.measurements`.

## Analysis semantics

`analysis.valid_measurements` includes EEA validity codes 1, 2, 3, and 4 and excludes
codes -1 and -99. Concentrations are retained as reported. City-hour means average all
available valid sampling-point values for a city, pollutant, and hour. Daily, monthly,
and annual means give each available city-hour mean equal weight. Missing measurements
are not imputed.

Calendar groupings use the fixed UTC+1 timestamps supplied by EEA, not daylight-saving
civil time. UTC columns subtract one hour. CO is reported in `mg.m-3`; the other five
pollutants use `ug.m-3`. Source coverage is preserved as-is: in particular, Rome has no
PM10 or PM2.5 rows in this snapshot. The five-city PM10 problem states that boundary
explicitly.

## Evaluation contract and identity

The problems cover annual trends and rankings, seasonal and diurnal patterns,
thresholds, multi-pollutant conditions, cross-city and cross-pollutant correlations,
episodes, station variation, coverage, validity, city dashboards, and provenance.
Expected answers are evaluator inputs and are not included in model prediction inputs.

The repository locator is
`evaluation-packs/eea-air-quality-six-cities-2018-2024-1.1.0.json`. It pins Hugging Face
commit `58958007cdf38eb9e563356f16ecd8011d5a3d67`, manifest SHA-256
`d2d29215ada662b7e6dcc70188a91ea9b7e50df8f77a20ee24521b869b137317`, case-export
SHA-256 `9f94d99fc98cac7a6cc9927e54b4f7887f2fac1cf052b6150743aacedcd57a34`, and database
SHA-256 `974df8df6bd3af538c1ca098f3c6c6bbe80f07e7a63ac565dc61c3ba6d7a3fd5`.
The loader verifies these identities before exposing the pack.

The 20 cases from `1.0.0` are preserved in `1.1.0`; the earlier release remains
available as immutable publication history.

## Source and license

The source is the [EEA Air Quality Download
Service](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d),
using its primary validated E1a data. The source data and this evaluation distribution
use the Creative Commons Attribution 4.0 International license. Attribution:
`European Environment Agency (EEA)`.

The release retains exact request bodies and source-archive SHA-256 digests. It is a
fixed snapshot rather than a live query; EEA does not endorse this derived evaluation
pack.

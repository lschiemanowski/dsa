# SMARD DE/LU 2024

The SMARD DE/LU 2024 release is DSA's second versioned evaluation pack. Version `1.2.0`
combines a content-pinned DuckDB database with one unified set of 100 exact-JSON
problems. Its neutral sequential identities do not encode a core/generalization split.

## Data

The database covers the 2024 `Europe/Berlin` calendar year for the `DE-LU` market area
at native 15-minute resolution, giving 35,136 interval rows. It combines six official
SMARD exports:

- realized electricity generation;
- realized grid and residual load;
- forecast grid and residual load;
- offshore-wind, onshore-wind, and photovoltaic forecasts;
- day-ahead electricity prices; and
- commercial net exports.

The recommended relations are `analysis.quarter_hourly` for interval analysis and
`analysis.daily` for local-calendar summaries. The database also retains raw relations
and source-request receipts in its `metadata` schema. Missing measurements are not
imputed.

## Evaluation contract

The problems cover generation mix, annual, monthly, daily, and interval aggregation;
load profiles and ramps; forecast accuracy, bias, and outliers; price distributions and
negative-price episodes; capture prices; imports and exports; system conditions;
calendar integrity; and provenance.
Their shared rules use `Europe/Berlin` for dates, UTC `Z` strings for interval
identities, the earliest result for ties, and forecast minus actual for forecast error.
Percentages are rounded to four decimal places, correlations to six, and other
calculated numbers to two, with halfway cases rounded away from zero.

The repository locator is
`evaluation-packs/smard-de-lu-2024-1.2.0.json`. It pins Hugging Face commit
`58958007cdf38eb9e563356f16ecd8011d5a3d67`, manifest SHA-256
`8afc8246308c445cb6e792ca344da66194826c79c744bdb824fb688ac7badea1`, case-export
SHA-256 `fa452ec4c64e6da08d3d68c1f769ef71ccf88b77b2974898808cba72f284777b`, and database
SHA-256 `249143dd8b399a6be84d666190f16b89ebc71545783362a50efaa40dc02bc45b`.

The 20 cases from `1.1.0` are preserved in `1.2.0`; the earlier release remains
available as immutable publication history.

The loader verifies the manifest, cases, and database before exposing the pack.
Expected answers remain evaluator inputs and are not included in model prediction
inputs.

## Source and license

The source is [SMARD market data](https://www.smard.de/) published by the German Federal
Network Agency. Attribution: `Bundesnetzagentur | SMARD.de`.

The source data and this evaluation distribution use the Creative Commons Attribution
4.0 International license. Historical SMARD data can change, so this pack uses the
checksum-pinned 2024 snapshot rather than live queries.

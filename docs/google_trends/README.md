# Google Trends Data Collection and Preprocessing

## 1. Purpose

This document describes the Google Trends component of the behavioral outcome data used in the age-verification policy project.

The purpose of the Google Trends data is to measure changes in public search behaviour related to VPN usage, circumvention, and attention to age-verification policies.

The final dataset covers four countries:

- United Kingdom (UK)
- Australia (AU)
- Ireland (IE)
- New Zealand (NZ)

The United Kingdom and Australia are the main treated countries, while Ireland and New Zealand are used as comparison/control countries.

The study period is:

**1 January 2025 to 30 June 2026**

## Repository locations

- Preprocessing notebook: `notebooks/google_trends/preprocessing.ipynb`
- Raw Google Trends CSV files: `data/`
- Processed daily series and scaling metadata: `data/processed/`

Run the notebook from the repository root. It reads raw CSV files without modifying them and writes the processed outputs under `data/processed/`.

---

## 2. Keyword Selection

Several candidate Google Trends queries were pilot-tested before the final data collection.

The initial candidate keywords were:

- `VPN`
- `free VPN`
- `age verification`
- `bypass age verification`
- `VPN age verification`
- `best VPN`

The purpose of the pilot screening was to determine whether each query produced sufficient daily variation across the four study countries.

### Final retained keywords

| Keyword | Role | Decision |
|---|---|---|
| `VPN` | Primary circumvention-related behavioral outcome | Keep |
| `free VPN` | Secondary / robustness circumvention proxy | Keep |
| `age verification` | Policy-attention indicator | Keep |
| `bypass age verification` | Direct bypass-related query | Drop |
| `VPN age verification` | Direct circumvention-related query | Drop |
| `best VPN` | Alternative VPN-intent query | Drop |

`VPN` was retained as the primary outcome because it produced the most continuous daily signal across all four countries.

`free VPN` was retained as a secondary indicator because it provided useful additional information about possible circumvention intent, especially in the UK and Australia.

`age verification` was retained as a policy-attention indicator rather than a direct circumvention measure. It showed strong responses around policy events in treated countries but was relatively sparse in some control countries.

The more specific long-tail queries, such as `bypass age verification` and `VPN age verification`, were excluded because they contained a high proportion of zero-valued daily observations and were therefore unsuitable for reliable daily time-series analysis.

---

## 3. Why Overlapping Windows Were Used

A single Google Trends query covering the full study period returned weekly rather than daily observations.

Because the analysis requires daily data, the full period was divided into shorter windows.

However, Google Trends independently normalizes each query to a scale from 0 to 100. Therefore, simply concatenating separately downloaded windows would produce inconsistent scales.

To solve this problem, overlapping windows were used.

The nine collection windows were:

| Window | Start date | End date |
|---|---|---|
| W01 | 2025-01-01 | 2025-03-31 |
| W02 | 2025-03-01 | 2025-05-31 |
| W03 | 2025-05-01 | 2025-07-31 |
| W04 | 2025-07-01 | 2025-09-30 |
| W05 | 2025-09-01 | 2025-11-30 |
| W06 | 2025-11-01 | 2026-01-31 |
| W07 | 2026-01-01 | 2026-03-31 |
| W08 | 2026-03-01 | 2026-05-31 |
| W09 | 2026-05-01 | 2026-06-30 |

Adjacent windows overlap by approximately one month.

For example:

```text
W01: Jan 2025 -------- Mar 2025
                         |
                         | March overlap
                         |
W02:                    Mar 2025 -------- May 2025

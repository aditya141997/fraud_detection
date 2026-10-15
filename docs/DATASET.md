# Dataset: IEEE-CIS Fraud Detection

Source: Kaggle competition [ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection), data
provided by Vesta Corporation. Access requires accepting the
[competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules). Raw files are not committed
(`data/` is gitignored).

Numbers in this file were measured on the local files with the profiling run described at the end.
Column meanings marked _(Kaggle)_ come from the competition's data description, not from this project's
analysis; Vesta masked most fields, so those meanings cannot be verified here.

## What is used

| File | Rows | Columns | Used |
|---|---|---|---|
| `data/raw/train_transaction.csv` | 590,540 | 394 | yes |
| `data/raw/train_identity.csv` | 144,233 | 41 | yes, left-joined on `TransactionID` |
| `test_transaction.csv`, `test_identity.csv` | – | – | no: the Kaggle test set has no labels |

After the join: **590,540 rows x 434 columns**, `TransactionID` unique (0 duplicates), about 2.7 GB in
memory as loaded by pandas. Only 24.4% of transactions (144,233) have an identity record; the rest have
all 40 identity columns null.

## Label

`isFraud` (0/1). 20,663 fraud rows = **3.50%** of transactions. Total fraud amount is 3,083,844.86.
The classes are heavily imbalanced, which is why the project reports PR-AUC and recall at fixed FPR
rather than accuracy.

Identity presence is itself informative: fraud rate is 7.85% where an identity record exists and 2.09%
where it does not.

Labels in the competition are chargeback-derived _(Kaggle)_. In production such labels arrive weeks
late; see [Known limitations](../README.md#known-limitations).

## Columns by group

| Group | Columns | Type | Avg. missing | What it is _(Kaggle)_ |
|---|---|---|---|---|
| Keys / target | `TransactionID`, `isFraud` | int | 0% | row id, label |
| Time | `TransactionDT` | int | 0% | seconds from an unknown reference time, not a real timestamp |
| Amount | `TransactionAmt` | float | 0% | transaction amount in USD |
| Product | `ProductCD` | cat | 0% | product code of the transaction |
| Card | `card1`-`card6` | mixed | 0-1.6% (train) | payment card info: type, category, issuer, country; masked |
| Address | `addr1`, `addr2` | float | 11.5% (train) | billing region and country |
| Distance | `dist1`, `dist2` | float | 61.9% / 92.9% (train) | distances, masked |
| Email | `P_emaildomain`, `R_emaildomain` | cat | 16.0% / 76.8% | purchaser and recipient email domain |
| Counting | `C1`-`C14` | float | 0% | counts such as addresses associated with the card; masked |
| Timedelta | `D1`-`D15` | float | 58.2% | days between previous transactions; masked |
| Match | `M1`-`M9` | cat (T/F) | 49.9% | match flags such as names on card and address |
| Vesta features | `V1`-`V339` | float | 43.0% | Vesta-engineered ranking, counting and entity relations; masked |
| Identity numeric | `id_01`-`id_11` | float | 73-99% (train) | identity and network signals; masked |
| Identity categorical | `id_12`-`id_38` | cat | 73-99% (train) | browser, OS, screen resolution, proxy flags, etc.; masked |
| Device | `DeviceType`, `DeviceInfo` | cat | 76% / 80% | device class and device string |

The competition treats these as categorical: `ProductCD`, `card1`-`card6`, `addr1`, `addr2`,
`P_emaildomain`, `R_emaildomain`, `M1`-`M9`, `DeviceType`, `DeviceInfo`, `id_12`-`id_38`. In this
project only pandas `object` columns (31 of them) are cast to LightGBM `category`; `card1`-`card3`,
`card5` and `addr*` stay numeric.

Column dtypes after the join: 399 float64, 31 object, 4 int64.

### Missingness

214 of 434 columns are more than 50% null and 12 are more than 90% null. Missingness is structural, not
random: the identity block is absent for 75.6% of rows, and many V columns are null in blocks together.
LightGBM handles nulls natively; the logistic-regression baseline median-imputes numeric columns and
drops the categorical ones.

## Key distributions

**Amount.** Mean 135.03, median 68.77, 75th percentile 125.00, max 31,937.39. Fraud rows: mean 149.24,
median 75.00, max 5,191.00. Fraud is not concentrated in large amounts, which is why a flat review cost
makes the cost-optimal threshold so low.

**Product code** (rows, fraud rate):

| ProductCD | Rows | Fraud rate |
|---|---|---|
| W | 439,670 | 2.04% |
| C | 68,519 | 11.69% |
| R | 37,699 | 3.78% |
| H | 33,024 | 4.77% |
| S | 11,628 | 5.90% |

**Card.** `card4` network: visa 384,767 (3.48%), mastercard 189,217 (3.43%), discover 6,651 (7.73%),
american express 8,328 (2.87%). `card6` type: debit 439,938 (2.43%), credit 148,986 (6.68%); "charge card"
and "debit or credit" have 15 and 30 rows, too few to read anything into.

**Cardinality.** `card1` 13,553 values, `card2` 500, `card3` 114, `card5` 119, `addr1` 332, `addr2` 74.

**Email.** 59 distinct purchaser domains, 60 recipient domains. The recipient domain is null for 76.8% of
rows.

**Device.** `DeviceType`: desktop 85,165, mobile 55,645, null 449,730. `DeviceInfo` is present for 118,666
rows with 1,786 distinct strings.

## Time

`TransactionDT` runs from 86,400 to 15,811,131 seconds, about **183 days (6 months)**. The project maps it
to timestamps with an arbitrary epoch (`2017-12-01`, `EPOCH` in [src/features.py](../src/features.py));
only ordering and gaps matter, so the calendar date is not meaningful.

## How this project splits it

Rows are sorted by `TransactionDT` and cut by time, never randomly ([src/data.py](../src/data.py)):

| Split | Rows | Fraud rate | Role |
|---|---|---|---|
| Train (oldest 70%) | 413,378 | 3.52% | fit models |
| Validation (next 15%) | 88,581 | 3.43% | tune hyperparameters, choose threshold |
| Replay (newest 15%) | 88,581 | 3.48% | final numbers; streamed through Redpanda |

Outputs: `data/processed/{train,val,replay}.parquet`, plus per-entity velocity features
`features_{card,device,email}.parquet`.

## Derived information

The dataset has no card ID, device ID or customer ID, so entities are constructed:

| Entity | Built from | Distinct values |
|---|---|---|
| card | `card1_card2_card3_card5` joined with `_` (nulls become `na`) | 14,845 |
| device | `DeviceInfo` | 1,786 |
| email domain | `P_emaildomain` | 59 |

Per entity, 4 past-only rolling features are computed (count and amount sum over 1h and 24h), 12 in total
across the three entities. Rows whose entity key is null get no entry. The window excludes the current
event; see [Feature contract](../README.md#feature-contract).

The card key is a proxy, not a real card number: different cardholders can share the same
`card1/2/3/5` combination, and one cardholder can appear under several. Device and email domain are weak
entities (shared by many people, null for most rows).

## Caveats about the data

- Anonymised features (`C*`, `D*`, `V*`, `id_*`) mean model explanations are correct but not
  human-readable.
- The data is one 6-month window from one merchant population. Drift seen between older and newer rows
  (logistic regression PR-AUC 0.385 on validation vs 0.167 on replay) may not generalise to other periods.
- Identity coverage is 24.4%, so any feature built from identity or device data describes a minority of
  transactions.
- Labels are treated as known immediately in training and evaluation.

## Reproducing these numbers

```powershell
make data     # merge, validate with Pandera, time-split -> data/processed/*.parquet
```

The figures above came from loading both CSVs, left-joining on `TransactionID`, and computing counts,
null rates and fraud rates directly with pandas. Group-level averages are over the columns in each group; the per-column ranges marked "(train)" were measured on the train split only.

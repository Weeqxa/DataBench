# DataBench development evaluation

Stage-one development evaluation; not an official competition submission

Selection: 100 paired questions; seed 42; dataset fingerprint `7cab82893c6a27eb`.

Percentages below use the official comparator. Conditional accuracy is accuracy among successful records.

| Metric | Full | Lite |
|---|---:|---:|
| total | 100 | 100 |
| successful | 97 | 95 |
| correct | 86 | 81 |
| overall_accuracy | 86.0 | 81.0 |
| conditional_accuracy | 88.6598 | 85.2632 |
| success_rate | 97.0 | 95.0 |
| processing_failure_rate | 3.0 | 5.0 |
| format_errors | 0 | 0 |

## Accuracy by answer type

| Type | Full correct / total | Lite correct / total |
|---|---:|---:|
| boolean | 19 / 20 | 18 / 20 |
| number | 16 / 20 | 17 / 20 |
| category | 16 / 20 | 14 / 20 |
| list[category] | 18 / 20 | 16 / 20 |
| list[number] | 17 / 20 | 16 / 20 |

## Paired comparison

- both_correct: 73; indices [989, 1602, 959, 1780, 1271, 900, 467, 795, 742, 660, 1008, 1275, 195, 863, 1267, 607, 716, 1280, 1110, 1038, 1261, 704, 168, 310, 1197, 537, 930, 471, 1579, 1390, 1219, 1185, 792, 1152, 255, 782, 282, 307, 532, 1278, 177, 164, 1557, 192, 33, 238, 1246, 1191, 112, 218, 1022, 1578, 851, 359, 361, 1248, 934, 859, 507, 753, 533, 823, 1370, 489, 921, 482, 1681, 1318, 397, 541, 27, 1016, 1306]
- full_only: 13; indices [678, 494, 967, 670, 515, 20, 1149, 919, 257, 680, 1131, 1365, 1000]
- lite_only: 8; indices [559, 83, 616, 947, 772, 210, 913, 1465]
- both_incorrect: 6; indices [592, 123, 1079, 57, 1256, 376]

## Error review

Manually reviewed: 33 / 33 errors.

Every failure is listed in errors.csv and errors.json. Categories inferred from explicit error messages only; wrong answers require manual review.
Do not infer missing tracebacks. Review data-selection and operation errors using the saved code and local table.

## Representative examples

### Index 559 (full, unknown)

List the diabetes outcomes of the top 5 individuals with the highest insulin levels.

Expected: `'[1, 1, 1, 1, 1]'`; predicted: `[1, 0, 0, 1, 1]`.

Verified local all.parquet: the five largest Insulin values are 846, 744, 680, 600, 579, with Outcome [1, 0, 0, 1, 1]. Generated code implements the literal question. Stored full reference is [1, 1, 1, 1, 1]; possible reference/table revision inconsistency, not a demonstrated model error.

### Index 592 (full, operation)

What is the most common number of programming languages used by participants on a regular basis?

Expected: `'2'`; predicted: `1`.

Verified language cells are NumPy arrays. Converting them to strings and splitting on commas counts most arrays as one item. Counting array lengths gives modal count 2 (5278 rows), matching the reference.

### Index 123 (full, operation)

What are the 5 highest counts of listings by a single host for entire homes?

Expected: `'[288, 288, 288, 288, 288]'`; predicted: `[288, 281, 260, 258, 224]`.

The generated groupby(host_id).max() collapses rows before selecting the top five. Direct row-wise nlargest gives [288, 288, 288, 288, 288], matching the reference; grouping removes the required repeated values.

### Index 1079 (full, data_selection)

Who is the author of the tweet with the most retweets?

Expected: `'Joe Biden'`; predicted: `None`.

Both attempts raised KeyError: retweets (saved traceback). Local columns include retweets<gx:number> and author_name<gx:category>, not retweets or author_name. The model discarded semantic suffixes from the supplied schema; the generic retry prompt did not repair the column name.

### Index 83 (full, unknown)

Identify the 5 largest total amounts paid for trips.

Expected: `'[80.0, 80.0, 80.0, 80.0, 80.0]'`; predicted: `[80.0, 80.0, 80.0, 80.0, 79.55]`.

Verified local all.parquet: total_amount.nlargest(5) is [80.0, 80.0, 80.0, 80.0, 79.55]. The code directly implements the question, but the reference expects five 80.0 values. Possible local-table/reference inconsistency; keep the official mismatch without declaring a model error.

### Index 57 (full, execution)

List the top 3 most common areas of knowledge.

Expected: `"['Computer Science', 'Business', 'Enginering & Architecture']"`; predicted: `None`.

Both saved tracebacks report NameError: pd is not defined. Generated code calls pd.notna without importing pandas; sandbox execution namespace contains only df. The repair prompt receives a generic failure and repeats the same missing import.

### Index 616 (full, operation)

Which company received the lowest rating?

Expected: `'N26'`; predicted: `'Wise'`.

The code chooses the first individual review with minimum rating (Wise), without aggregating by company. Both companies have rating-1 reviews; mean ratings are Wise=4.60078125 and N26=3.635645933014354, consistent with the reference N26 under a company-level interpretation. The question does not explicitly specify averaging, so retain that ambiguity.

### Index 947 (full, data_selection)

Who are the 6 customers with the highest total transaction amount?

Expected: `'[18484, 17744, 17634, 17498, 17437, 17350]'`; predicted: `[9667, 6883, 5578, 4026, 5500, 1160]`.

The code returns customer id values [9667, 6883, 5578, 4026, 5500, 1160]. The reference instead contains the six highest total_trans_amt values, verified as [18484, 17744, 17634, 17498, 17437, 17350]. The wording Who are the customers makes the identifier interpretation plausible; document this output-field ambiguity.

### Index 772 (full, data_selection)

Are there any married individuals who reflected on 'exercise'?

Expected: `'True'`; predicted: `False`.

The code compares reflection_period with exercise. That column specifies the recall interval rather than the content of the happy moment. It should inspect the moment text or its topic for married respondents; the selected column makes the predicate false.

### Index 913 (full, data_selection)

How many unique teams are there in the dataset?

Expected: `'230'`; predicted: `1184`.

Verified Team.nunique() is 1184, whereas NOC.nunique() is 230, matching the reference. The code uses the literal Team column; the reference appears to define teams by country/NOC rather than named team entries. Output-column/definition ambiguity, not an execution failure.

## Team handoff

- Participant 1: reproduce failures using saved generated code and logs; any baseline fix requires a new experiment.
- Participant 3: use this table, paired comparison and reviewed examples; these are development scores, not competition scores.
- Participant 4: use config.json, execution_environment.json and frozen question metadata to verify reproducibility.

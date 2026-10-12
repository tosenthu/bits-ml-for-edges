# Task 1 submission package

**Student:** Senthil Kumar M R  
**Student ID:** 2025cs05049

## What to upload

The main Task 1 deliverable is `Task1_Report.md`. It contains the measured results, the edge/cloud comparison, and answers to reflection questions R1–R5.

The course worksheet asks for the results to be consolidated in a lab report. It does not explicitly require every model binary or raw terminal log as a separate upload. Keep the rest of this package as supporting material, and attach it only if the submission portal accepts source/evidence files or an archive.

## What is included

- `Task1_Report.md` — the report with measured results, the edge/cloud comparison, and answers to reflection questions R1–R5.
- `code/` — the six worksheet programs with descriptive filenames, plus helper scripts for the SavedModel size measurement and full-precision timing audit.
- `supporting/task1-results.json` — machine-readable summary of the run.
- `supporting/logs/` — captured console output and timing samples used to check the reported values.
- `supporting/models/` — the trained Keras model, TFLite model, benchmark sample, and SavedModel directory used for the size calculation.

The source and logs are included for reproducibility. The report is the document-style submission artifact. This package does not contain fabricated screenshots; the Task 1 worksheet does not specify a screenshot requirement.

## Before submitting

1. Read the report and make any changes needed to reflect your own understanding and observations.
2. Check whether the Taxila portal accepts Markdown files. If not, ask the course team what format they accept; this package intentionally does not convert the report to DOCX.
3. If uploading an archive, include this README so the supporting files are clear.

The timing run used a MacBook Pro with an Apple M3 Pro processor and simulated the edge device on the laptop CPU. The cloud-style server also ran locally over `localhost`; it was not a remote cloud benchmark.

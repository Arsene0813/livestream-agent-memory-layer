# Period Conventions

This project uses ISO-style period fields for retail operations data.

## Monthly rows

Monthly data should use all three fields:

- `period_month`: month label in `YYYY-MM` format
- `period_start`: first calendar day of the period in `YYYY-MM-DD` format
- `period_end`: last calendar day of the period in `YYYY-MM-DD` format

Example:

period_month = 2026-03  
period_start = 2026-03-01  
period_end = 2026-03-31

## Range-level facts

Range-level facts should use:

- `period_granularity`
- `period_start`
- `period_end`
- `period_label`

Example:

period_granularity = month_range  
period_start = 2026-02-01  
period_end = 2026-04-30  
period_label = 2026-02_to_2026-04

## Narrative text

Narrative documents may use human-readable month names, but they should include the ISO date range when referring to the demo period.

Preferred format:

February–April 2026 (2026-02-01 to 2026-04-30)

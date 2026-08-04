# Dataset notes

The uploaded CSV contains 10,000 item-level rows covering 367 calendar dates, 50 restaurant IDs, and 14 menu items.

The file does not contain a customer `order_id`. It therefore supports demand/quantity forecasting but cannot directly establish the number of customer transactions.

The preprocessing pipeline:

1. Sums `quantity_sold` for each restaurant and date.
2. Averages those restaurant-level totals for each date.
3. Uses that average as a single-restaurant-style daily demand target.
4. Creates lag and rolling features using only earlier dates.

The source and licence of the uploaded CSV were not embedded in the file. Record the original download page, author, licence, and download date in the project report before presenting it as a research dataset.

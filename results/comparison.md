Out-of-sample window: 2025-05-15 15:00:00-04:00 to 2026-11-19 11:00:00-05:00, 8735 bars, 392 trading days (about 1.6 years). Sharpe is from daily USD P&L for one contract, annualised with sqrt(252), risk-free zero. Intervals are 95 percent circular block-bootstrap intervals (2000 paths, 5-day blocks). With this few days they are wide.

### Run: main (commission 0.00 USD per side, 7.50 USD per side per contract in total, ASSUMPTIONS)

| Strategy | H | Gross Sharpe [95% CI] | Net Sharpe [95% CI] | Sides | Sides/day | Net P&L/yr USD | Max DD USD | Hit rate | Break-even cost x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum | 1 | -0.60 [-1.89, 0.75] | -0.67 [-1.96, 0.69] | 1401 | 3.57 | -69,356 | 142,285 | 0.49 | 0.00 |
| momentum | 2 | -0.35 [-1.59, 1.09] | -0.40 [-1.64, 1.04] | 993 | 2.53 | -37,911 | 141,620 | 0.49 | 0.00 |
| buy-and-hold, tradable samples (like-for-like) | 1 | 0.21 [-1.30, 1.68] | 0.21 [-1.30, 1.68] | 1 | 0.00 | 22,566 | 132,105 | 0.54 | 361.03 |
| buy-and-hold, tradable samples (like-for-like) | 2 | 0.16 [-1.37, 1.61] | 0.16 [-1.37, 1.61] | 1 | 0.00 | 15,967 | 131,570 | 0.53 | 255.74 |
| buy-and-hold, passive (reference) | n/a | 0.67 [-0.81, 2.09] | 0.67 [-0.81, 2.08] | 1 | 0.00 | 71,317 | 97,940 | 0.54 | 1138.82 |

Net Sharpe at 0x, 1x, 2x, 3x of the assumed costs:

| Strategy | H | 0x | 1x | 2x | 3x |
| --- | --- | --- | --- | --- | --- |
| momentum | 1 | -0.60 | -0.67 | -0.73 | -0.79 |
| momentum | 2 | -0.35 | -0.40 | -0.45 | -0.51 |
| buy-and-hold, tradable samples (like-for-like) | 1 | 0.21 | 0.21 | 0.21 | 0.21 |
| buy-and-hold, tradable samples (like-for-like) | 2 | 0.16 | 0.16 | 0.16 | 0.16 |
| buy-and-hold, passive (reference) | n/a | 0.67 | 0.67 | 0.67 | 0.67 |

Momentum minus like-for-like buy-and-hold, net Sharpe difference (paired bootstrap):
- H=1: -0.88 [-3.02, 1.38]
- H=2: -0.56 [-2.48, 1.56]

### Run: conservative (commission 2.50 USD per side, 10.00 USD per side per contract in total, ASSUMPTIONS)

| Strategy | H | Gross Sharpe [95% CI] | Net Sharpe [95% CI] | Sides | Sides/day | Net P&L/yr USD | Max DD USD | Hit rate | Break-even cost x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum | 1 | -0.60 [-1.89, 0.75] | -0.69 [-1.98, 0.67] | 1401 | 3.57 | -71,627 | 145,375 | 0.49 | 0.00 |
| momentum | 2 | -0.35 [-1.59, 1.09] | -0.42 [-1.66, 1.02] | 993 | 2.53 | -39,526 | 142,810 | 0.49 | 0.00 |
| buy-and-hold, tradable samples (like-for-like) | 1 | 0.21 [-1.30, 1.68] | 0.21 [-1.30, 1.68] | 1 | 0.00 | 22,545 | 132,115 | 0.54 | 270.77 |
| buy-and-hold, tradable samples (like-for-like) | 2 | 0.16 [-1.37, 1.61] | 0.16 [-1.37, 1.61] | 1 | 0.00 | 15,946 | 131,580 | 0.53 | 191.81 |
| buy-and-hold, passive (reference) | n/a | 0.67 [-0.81, 2.09] | 0.67 [-0.81, 2.08] | 1 | 0.00 | 71,296 | 97,940 | 0.54 | 854.12 |

Net Sharpe at 0x, 1x, 2x, 3x of the assumed costs:

| Strategy | H | 0x | 1x | 2x | 3x |
| --- | --- | --- | --- | --- | --- |
| momentum | 1 | -0.60 | -0.69 | -0.77 | -0.86 |
| momentum | 2 | -0.35 | -0.42 | -0.49 | -0.56 |
| buy-and-hold, tradable samples (like-for-like) | 1 | 0.21 | 0.21 | 0.21 | 0.21 |
| buy-and-hold, tradable samples (like-for-like) | 2 | 0.16 | 0.16 | 0.16 | 0.16 |
| buy-and-hold, passive (reference) | n/a | 0.67 | 0.67 | 0.67 | 0.67 |

Momentum minus like-for-like buy-and-hold, net Sharpe difference (paired bootstrap):
- H=1: -0.90 [-3.04, 1.36]
- H=2: -0.58 [-2.49, 1.54]

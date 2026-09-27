# Tactical Evolution of the WSL: Pre-COVID → COVID → Post-Boom

**Did the FA Women's Super League actually play differently once fans came
back and once the post-Euro 2022 investment wave hit?**

This project uses full-event StatsBomb data (open data, courtesy of Hudl
StatsBomb) to test whether WSL team tactics and performance shifted across
three distinct eras, using statistical testing, unsupervised clustering, and
a supervised classifier then digs into the league's top 4 teams specifically with shot maps, xG trendlines, and radar comparisons.

## TL;DR result

A Random Forest trained to guess which era a team-season came from, using
only 13 tactical/performance stats, hits **48.6% accuracy** (Leave-One-Out
CV) against a **33.3% chance baseline**, and that gap is statistically real
(permutation test **p = 0.005**, n=200 permutations). The signal isn't
coming from shot quality or xG, those are flat across all three eras, it's
coming from **how teams move the ball and defend**:

| Feature | 2018/19 | 2020/21 (COVID) | 2023/24 | ANOVA p |
|---|---|---|---|---|
| Avg. pass length | 22.4m | 21.6m | **20.0m** | 0.0001 |
| Pressures / 90 | 159 | 161 | **196** | 0.0003 |
| Long ball % | 23.1% | 21.5% | **17.2%** | 0.0004 |
| Pass completion % | 70.0% | 73.0% | **78.1%** | 0.028 |
| xG / 90 | 1.40 | 1.45 | 1.49 | 0.95 (n.s.) |
| Possession % | ~50% | ~50% | ~50% | 1.00 (n.s.) |

**Reading:** the post-2022 WSL isn't scoring more or dominating possession
more. It's pressing much harder (+23% pressures/90) and playing shorter,
more accurate, more controlled possession (pass length down ~2.4m,
completion up 8 points). That's consistent with the professionalization
story (better fitness/conditioning, more coaching investment, more athletic
squads) rather than a simple "crowds are back" effect, the shift keeps
growing from 2020/21 to 2023/24, well past the point fans returned.

See `outputs/era_comparison_boxplots.png` and
`outputs/feature_importance.png` for the visual evidence.

## Top 4 teams, up close

On top of the league-wide era comparison, `src/plot_top4.py` pulls the
2023/24 points table from match results and drills into the top 4 finishers
(currently Manchester City, Chelsea, Arsenal, and Liverpool):

- **`outputs/top4_shotmaps.png`** — xG-weighted shot maps for each of the
  top 4, goals highlighted, to see whether the league's best sides differ in
  shot *selection* as well as shot *volume*.
- **`outputs/top4_xg_trend.png`** — rolling 5-match xG per team across the
  2023/24 season, to see title-race form swings rather than just a season
  total.
- **`outputs/top4_radar.png`** — each top-4 team's tactical profile
  (possession, passing, pressing) plotted against the league average on one
  radar, to see how the best teams deviate from the pack.

## Why this dataset / question

Hudl StatsBomb's free open data release (May 2026) made full 2023/24
event data (passes, carries, shots, pressures, and their proprietary
xG model) available for the WSL for the first time, on top of the
pre-existing 2018/19, 2019/20 and 2020/21 open releases. That gives a rare
same-schema, same-competition dataset spanning a genuine natural experiment:
football played with fans (2018/19), football played in empty COVID-era
stadiums with a compressed fixture list (2020/21), and football played
post-Euro-2022-boom with record investment and attendance (2023/24).

## Methodology

1. **`src/collect_data.py`** — pulls all WSL matches and full event data
   for the three target seasons via `statsbombpy` (370 matches, ~1.3M
   events total).
2. **`src/build_features.py`** — aggregates raw events into 13 team-season
   features covering possession, passing style, pressing intensity, and
   shot/xG quality, normalized to per-90 rates. Includes an explicit fix
   for a StatsBomb open-data quirk where two teams (Aston Villa,
   Manchester United) are tagged with two different name strings across
   matches in the 2020/21 season — confirmed by diffing event-level team
   names against the match sheet, not assumed.
3. **`src/analyze.py`** — the core league-wide analysis:
   - **One-way ANOVA** per feature across the three eras, to identify which
     stats actually differ significantly rather than relying on eyeballed
     bar charts.
   - **PCA + KMeans clustering**, blind to era labels, to check whether
     team-seasons naturally separate by era.
   - **Random Forest classifier** predicting era from tactical stats,
     evaluated with **Leave-One-Out cross-validation** (appropriate given
     n=35) and a **permutation test** to confirm the accuracy is real
     signal and not overfitting on a small sample
4. **`src/plot_top4.py`** — top-4-team drilldown for the 2023/24 season:
   - Computes a points table directly from match results to identify the
     top 4 finishers, rather than hardcoding team names.
   - Shot maps (via `mplsoccer`), rolling xG trendlines, and a
     league-average-vs-top-4 radar chart, all built from the same event
     data and feature table as the league-wide analysis.

## Honest limitations

- **n=35 team-seasons** (11–12 teams × 3 seasons). This is small. LOO-CV
  and the permutation test exist specifically to guard against
  small-sample overfitting claims, read the accuracy number as "there is
  real signal here," not as "this model is production-grade."
- This is **observational**, not causal. The post-2022 shift is correlated
  with the investment/popularity boom, but I can't rule out other
  confounds (rule changes, squad turnover, general league-wide tactical
  trends unrelated to the WSL specifically, etc.) with this data alone.
- 2019/20 (the COVID-truncated season) was excluded from the 3-way
  comparison to keep a clean pre/during/post structure; it's still pulled
  by `collect_data.py` if you want to extend the analysis.
- The top-4 standings table is computed from match results only (3/1/0
  points), so it reflects final league position for 2023/24, not a
  reconstruction of the exact live table at any point mid-season.

## Reproducing this

```bash
pip install -r requirements.txt
python src/collect_data.py     # ~5-8 min, pulls ~1.3M events from StatsBomb open data
python src/build_features.py   # builds data/team_season_features.csv
python src/analyze.py          # runs all stats/ML, writes core outputs to outputs/
python src/plot_top4.py        # top-4 shot maps, xG trend, and radar
```

## Repo structure

```
src/
  collect_data.py      # StatsBomb open data pull (statsbombpy)
  build_features.py    # raw events -> team-season feature table
  analyze.py            # ANOVA, PCA/KMeans, Random Forest + permutation test
  plot_top4.py          # top-4 shot maps, xG trendlines, radar vs league average
data/
  team_season_features.csv   # final feature table (35 rows x 13 features)
outputs/
  anova_results.csv
  cluster_assignments.csv
  classifier_report.txt
  era_comparison_boxplots.png
  feature_importance.png
  pca_clusters.png
  top4_shotmaps.png
  top4_xg_trend.png
  top4_radar.png
```

## Data source & attribution

Data: [Hudl StatsBomb Open Data](https://github.com/statsbomb/open-data),
accessed via [`statsbombpy`](https://github.com/statsbomb/statsbombpy).
Per StatsBomb's terms, any research or analysis based on this data credits
StatsBomb as the source.
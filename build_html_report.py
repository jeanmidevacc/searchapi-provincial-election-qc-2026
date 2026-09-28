"""Build the HTML briefing report from the Kaggle dgeq/ export.

Reads data/kaggle/dgeq/ -- run import_from_kaggle.py first (see readme.md).

Uses the `briefing` package (https://pypi.org/project/briefing/).
"""

# =============================================================================
# Setup
# =============================================================================
import base64
import gzip
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import tldextract

import briefing as bf

from markdown_it import MarkdownIt

from chatgpt_embeddings import embed_answers

plt.style.use("ggplot")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "kaggle" / "dgeq"
SEARCHAPI_DIR = HERE / "data" / "kaggle" / "searchapi"
OUT_PATH = HERE / "reports" / "report.html"


# =============================================================================
# Load DGEQ data (from the Kaggle dgeq/ export -- already cleaned/joined by
# build_circonscriptions.py / build_electeurs_candidatures.py upstream)
# =============================================================================
# District list + boundaries, already joined 1:1 on `code`.
circonscriptions = pd.read_csv(DATA_DIR / "circonscriptions_2026.csv")
districts = circonscriptions[["code", "circonscription", "superficie_km2_approx"]]

# Registered electors per district.
electors = pd.read_csv(DATA_DIR / "electeur_inscrit.csv")
electors = electors.rename(columns={"CODE_CIRCONSCRIPTION": "code", "NOMBRE_ELECTEURS_AU_DECRET": "n_electors"})

# Candidacies -- one row per candidate; count per district.
candidatures = pd.read_csv(DATA_DIR / "candidatures.csv")
n_candidates = candidatures.groupby("code_circonscription").size().rename("n_candidates")

# District boundaries -- one geometry (GeoJSON string) per row; rebuild a
# FeatureCollection keyed by `code` (as each feature's top-level "id") so it
# lines up with the "code" column used everywhere else below.
geo = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "id": row.code, "geometry": json.loads(row.geometry_geojson)}
        for row in circonscriptions.itertuples()
    ],
}

# One row per district: identity + electors + candidacies. Left-joined off
# `districts` (127 rows, the authoritative list) so a district with no
# candidacies recorded yet still gets a row instead of silently disappearing.
df = (
    districts[["code", "circonscription", "superficie_km2_approx"]]
    .merge(electors[["code", "n_electors"]], on="code", how="left")
    .merge(n_candidates, left_on="code", right_index=True, how="left")
)
df["n_candidates"] = df["n_candidates"].fillna(0).astype(int)
df["electors_per_km2"] = df["n_electors"] / df["superficie_km2_approx"]


# =============================================================================
# Chart 1 -- candidates per party (matplotlib, ggplot style)
# =============================================================================
# Quebec's National Assembly has 127 seats -- a party needs a candidate in at
# least this many districts to even have a mathematical shot at a majority.
SEATS_TOTAL = 127
MAJORITY_SEATS = SEATS_TOTAL // 2 + 1

party_counts = (
    candidatures.dropna(subset=["abreviation_parti"])
    .groupby(["abreviation_parti", "nom_parti"])
    .size()
    .rename("n_candidates")
    .reset_index()
    .sort_values("n_candidates")
)

fig_candidates, ax_candidates = plt.subplots(figsize=(8, 7))
ax_candidates.barh(party_counts["nom_parti"], party_counts["n_candidates"], color="#2a78d6")
ax_candidates.axvline(MAJORITY_SEATS, color="black", ls="--", lw=1.5)
ax_candidates.text(MAJORITY_SEATS, -0.7, f" {MAJORITY_SEATS} candidacies = majority cut-off",
                    ha="left", va="top", fontsize=8, fontweight="bold")
ax_candidates.set(title="Number of candidates per party", xlabel="candidates")
ax_candidates.tick_params(axis="y", labelsize=8)
plt.tight_layout()


# =============================================================================
# Chart 2 -- elector density per circonscription (plotly geo map)
# =============================================================================
# Electors per km^2 rather than raw elector count, so a district's color
# reflects how crowded it is rather than mostly just how big it is. Density
# spans ~5 orders of magnitude here (Ungava: 0.04/km^2 vs Mercier: 8344/km^2),
# so it's log-scaled for color -- a linear scale would paint almost every
# district the same flat color except the handful of dense urban ridings.
# Diverging red/blue around the median district turns that into a genuine
# polarity: redder = denser than a typical district, bluer = sparser.
df["log_density"] = np.log10(df["electors_per_km2"])
median_log_density = df["log_density"].median()

# The sparse (rural) arm stretches much further from the median than the dense
# (urban) arm does -- min-to-median is ~4 log units, median-to-max only ~1.5.
# Coloring across the *full* asymmetric span means most districts sit well
# short of either pole and read as pale gray. Clip the color domain to a
# symmetric radius (the shorter of the two arms) so both sides actually reach
# full saturation; the handful of most extreme districts (e.g. Ungava) just
# clip to solid blue/red instead of desaturating everything else.
half_span = min(median_log_density - df["log_density"].min(),
                 df["log_density"].max() - median_log_density)

DIVERGING_BLUE = "#2a78d6"   # sparser than the median district
DIVERGING_GRAY = "#f0efec"   # ~ the median district
DIVERGING_RED = "#e34948"    # denser than the median district

fig_electors = px.choropleth_map(
    df,
    geojson=geo,
    locations="code",
    color="log_density",
    hover_name="circonscription",
    hover_data={"log_density": False, "electors_per_km2": ":,.0f", "code": False},
    color_continuous_scale=[[0, DIVERGING_BLUE], [0.5, DIVERGING_GRAY], [1, DIVERGING_RED]],
    range_color=[median_log_density - half_span, median_log_density + half_span],
    color_continuous_midpoint=median_log_density,
    labels={"electors_per_km2": "Electors / km2"},
    map_style="carto-positron",
    # Centered on the Bertrand riding (Laurentides), between the two cities,
    # so Montreal falls near the bottom of the view and Quebec City near the
    # top right -- covers both while still leaving the tightly-packed
    # Montreal ridings (Mercier, Gouin, Laurier-Dorion, ...) legible. Still an
    # interactive slippy map, so the viewer can pan/zoom from here.
    center={"lat": 46.19, "lon": -74.12},
    zoom=5.6,
    opacity=1.0,
)

# Colorbar in real electors/km2, not log10 units.
tick_values = [0.1, 1, 10, 100, 1000, 8000]
fig_electors.update_layout(
    title="Electors per km2 (log scale, diverging from the median circonscription)",
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_colorbar=dict(
        title="Electors/km2",
        tickvals=[np.log10(v) for v in tick_values],
        ticktext=[f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in tick_values],
    ),
)


# =============================================================================
# Load SearchApi data (Google News + Google Rank Tracking) -- from the same
# Kaggle export's searchapi/ folder, see data/kaggle/searchapi/DATA_DICTIONARY.md
# =============================================================================
google_news = pd.read_csv(SEARCHAPI_DIR / "google_news.csv", low_memory=False)
google_rank = pd.read_csv(SEARCHAPI_DIR / "google_rank_tracking.csv", low_memory=False)

# party_key here matches configuration_party.json's keys 1:1 (verified against
# candidatures.csv's abreviation_parti too) -- reuse it for display labels.
party_config = json.loads((DATA_DIR / "configuration_party.json").read_text(encoding="utf-8"))
PARTY_LABEL = {k: v["fr_short_title"] for k, v in party_config.items()}

# The 5 parties that declared a candidate for premier get emphasis in both
# charts below -- 16 in-election parties is too many for individually
# distinct identity colors, so these 5 "headline" parties share one accent
# color and the other 11 share one neutral grey (same convention as
# analysis/config/party_style.yaml's `emphasis`, reused here verbatim).
HEADLINE_PARTIES = {k for k, v in party_config.items() if v.get("candidate_prime_minister")}
ACCENT_COLOR = "#2a78d6"  # headline party
OTHER_COLOR = "#b6b5b0"   # other in-election party

# Fixed fr/en display order for the 5 headline parties, each in that party's
# own real brand hue -- unlike analysis/config/party_style.yaml's
# `identity_colors` (a deliberately non-authentic CVD-safe assignment, since
# CAQ/PQ/PCOQ are all blue-branded in real life -- see that file's D4
# comment), this report intentionally uses each party's actual color family.
# CAQ/PQ/PLQ/PCOQ/QS hues sourced from Wikipedia's Canadian-party-colour
# infobox data (Template:Canadian party colour/colour, en.wikipedia.org,
# 2026-09), then each snapped in OKLCH -- hue held, lightness/chroma nudged
# into a passing band -- via the dataviz skill's validate_palette.js so the
# 3 blues (CAQ/PQ/PCOQ) still clear both the adjacent-pair check (line/bar
# charts) and the all-pairs check (the 3D answer map and network graphs
# below, where any two party dots can end up next to each other). All 5
# clear both; `validate_palette.js "1E90FF,00cbde,d7506e,3d50a0,ee9733"
# --mode light --pairs all` reproduces the check.
IDENTITY_COLORS = {
    "ÉCF-CAQ": "#1E90FF",
    "PQ": "#00cbde",
    "PLQ/QLP": "#d7506e",
    "PCOQ": "#3d50a0",
    "QS": "#ee9733",
}


# =============================================================================
# Chart 3 -- daily unique Google News articles per party, all 16 compared
# (plotly line chart -- interactive legend to isolate/compare series)
# =============================================================================
# "Unique" = de-duplicated per (party, link) -- a party's party-name and
# leader-name queries can both surface the same article. Bucketed by the
# article's own publish date (iso_date), not collection date -- but almost
# everything before the collector's own go-live is old evergreen/syndicated
# content Google still surfaces on a fresh query (14-100 rows/year back to
# 2004) rather than real day-to-day news flow, so it's dropped here: it would
# flatten the actual collection window into an unreadable sliver on the right
# edge. `scheduled_news` (collecter/app.py) started its every-12h sweeps
# 2026-09-07 -- see collecter/readme.md -- so that's the real start of
# meaningful day-over-day signal in this dataset, not an arbitrary cutoff.
COLLECTION_START = "2026-09-07"
news_dated = google_news[google_news["target_type"].isin(["party", "leader"])].copy()
news_dated["date"] = pd.to_datetime(news_dated["iso_date"], utc=True, errors="coerce").dt.floor("D")
news_dated = news_dated[news_dated["date"] >= COLLECTION_START]
news_dated = news_dated.drop_duplicates(subset=["party_key", "link"])

daily_counts = (
    news_dated.groupby(["party_key", "date"]).size().rename("n").reset_index()
    .pivot(index="date", columns="party_key", values="n")
)
# Reindex to every calendar day in range so a silent day plots as a real 0,
# not a skipped point that draws a misleading straight line across the gap.
full_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
daily_counts = daily_counts.reindex(full_range).fillna(0).astype(int)

fig_news = go.Figure()

# Other 11 parties first (drawn underneath) -- thin, semi-transparent grey,
# individually hoverable but sharing one legend entry via a dummy trace +
# groupclick, per the emphasis convention (identity here is not the point).
for party_key in daily_counts.columns:
    if party_key in IDENTITY_COLORS:
        continue
    fig_news.add_trace(go.Scatter(
        x=daily_counts.index, y=daily_counts[party_key],
        mode="lines", name=PARTY_LABEL[party_key],
        line=dict(color=OTHER_COLOR, width=1),
        opacity=0.55, legendgroup="other", showlegend=False,
        hovertemplate="%{fullData.name}<br>%{x|%b %d}: %{y} unique articles<extra></extra>",
    ))
fig_news.add_trace(go.Scatter(
    x=[None], y=[None], mode="lines", name="Other in-election parties (11)",
    line=dict(color=OTHER_COLOR, width=1.5),
    legendgroup="other", showlegend=True, hoverinfo="skip",
))

# Headline parties last (drawn on top), each its own identity color.
for party_key, color in IDENTITY_COLORS.items():
    fig_news.add_trace(go.Scatter(
        x=daily_counts.index, y=daily_counts[party_key],
        mode="lines", name=PARTY_LABEL[party_key],
        line=dict(color=color, width=2.5),
        hovertemplate="%{fullData.name}<br>%{x|%b %d}: %{y} unique articles<extra></extra>",
    ))

window_start = daily_counts.index.min().strftime("%b %d, %Y")
window_end = daily_counts.index.max().strftime("%b %d, %Y")
fig_news.update_layout(
    title=f"Daily unique Google News articles per party ({window_start} - {window_end})",
    xaxis_title="date (article's own publish date)",
    yaxis_title="unique articles / day",
    legend=dict(groupclick="togglegroup"),
    margin=dict(l=0, r=0, t=40, b=0),
)


# =============================================================================
# Chart 4 -- leader -> top-ranked domain network (plotly node-link diagram)
# =============================================================================
# Domain categories, reused for the edge/node colors below.
GOV_DOMAIN_MARKERS = ("electionsquebec.qc.ca", "assnat.qc.ca", ".gouv.qc.ca", "elections.ca")
SOCIAL_DOMAINS = {
    "www.facebook.com", "www.youtube.com", "www.instagram.com",
    "x.com", "twitter.com", "www.linkedin.com", "ca.linkedin.com",
    "www.reddit.com", "www.tiktok.com",
}


def categorize_domain(domain: str) -> str:
    if any(marker in domain for marker in GOV_DOMAIN_MARKERS):
        return "Government / official"
    if "wikipedia.org" in domain:
        return "Wikipedia"
    if domain in SOCIAL_DOMAINS:
        return "Social media"
    return "News & other websites"


CATEGORY_COLOR = {
    "Government / official": "#2a78d6",
    "Wikipedia": "#1baf7a",
    "Social media": "#eb6834",
    "News & other websites": "#b6b5b0",
}

# For each leader, "top rank" = the domain that showed up at position 1 most
# often across every fr/en sweep -- the site Google actually puts first when
# someone searches that leader's name. fr and en are kept separate (a
# leader's French results can top out on their own party site while English
# tops out on Wikipedia) then collapsed into one edge when they agree.
leader_top1 = (
    google_rank[(google_rank["target_type"] == "leader") & (google_rank["position"] == 1)]
    .dropna(subset=["domain"])
)
domain_modes = (
    leader_top1.groupby(["party_key", "lang", "domain"]).size().rename("n").reset_index()
    .sort_values("n", ascending=False)
    .groupby(["party_key", "lang"]).first()  # most-frequent #1 domain per (leader, lang)
    .reset_index()
)
edges = (
    domain_modes.groupby(["party_key", "domain"])["lang"]
    .apply(lambda s: "/".join(sorted(s)))
    .reset_index(name="langs")
)
edges["category"] = edges["domain"].map(categorize_domain)

# Bipartite layout: leaders on the left (headline parties first, same order
# as Chart 3), domains on the right ranked by how many leaders they're the
# top result for -- so shared hubs like Wikipedia surface near the top.
leader_order = list(IDENTITY_COLORS) + sorted(
    k for k in edges["party_key"].unique() if k not in IDENTITY_COLORS
)
domain_order = edges["domain"].value_counts().sort_values(ascending=False).index.tolist()
n_leaders, n_domains = len(leader_order), len(domain_order)

leader_y = {party_key: n_leaders - 1 - i for i, party_key in enumerate(leader_order)}
# Stretch the (shorter) domain list to span the same vertical range as the
# leader list, so edges don't all crowd into one corner.
domain_y = {
    domain: (n_domains - 1 - i) * (n_leaders - 1) / max(n_domains - 1, 1)
    for i, domain in enumerate(domain_order)
}

fig_rank = go.Figure()

# One edge trace per domain category -- legend reads as up to 4 categories,
# not 25 individual lines.
for category, color in CATEGORY_COLOR.items():
    cat_edges = edges[edges["category"] == category]
    if cat_edges.empty:
        continue
    xs, ys, hover = [], [], []
    for row in cat_edges.itertuples():
        xs += [0, 1, None]
        ys += [leader_y[row.party_key], domain_y[row.domain], None]
        label = f"{PARTY_LABEL[row.party_key]} -> {row.domain} ({row.langs})"
        hover += [label, label, None]
    fig_rank.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=category,
        line=dict(color=color, width=1.5), opacity=0.75,
        hovertext=hover, hoverinfo="text",
    ))

# Leader nodes -- headline parties in their identity color (Chart 3's
# convention), the other 11 in the shared neutral grey.
fig_rank.add_trace(go.Scatter(
    x=[0] * n_leaders, y=[leader_y[p] for p in leader_order],
    mode="markers+text", text=[PARTY_LABEL[p] for p in leader_order],
    textposition="middle left", textfont=dict(size=11),
    marker=dict(size=10, color=[IDENTITY_COLORS.get(p, OTHER_COLOR) for p in leader_order]),
    hovertemplate="%{text}<extra></extra>", showlegend=False,
))
# Domain nodes.
fig_rank.add_trace(go.Scatter(
    x=[1] * n_domains, y=[domain_y[d] for d in domain_order],
    mode="markers+text", text=domain_order,
    textposition="middle right", textfont=dict(size=11),
    marker=dict(size=7, color="#52514e"),
    hovertemplate="%{text}<extra></extra>", showlegend=False,
))

fig_rank.update_layout(
    title="Party leaders -> their #1-ranked Google search result",
    xaxis=dict(visible=False, range=[-0.9, 2.3]),
    yaxis=dict(visible=False),
    margin=dict(l=10, r=180, t=40, b=10),
    height=750,
)


# =============================================================================
# Load Google Trends data -- query-based Trends only covers the 5 headline
# ("candidate_prime_minister") parties (see DECISIONS.md), so every chart
# below is naturally scoped to those 5, using the same IDENTITY_COLORS.
# =============================================================================
trends_trending_now = pd.read_csv(SEARCHAPI_DIR / "google_trends_trending_now.csv", low_memory=False)
trends_timeseries = pd.read_csv(SEARCHAPI_DIR / "google_trends_timeseries.csv", low_memory=False)
trends_related_topics = pd.read_csv(SEARCHAPI_DIR / "google_trends_related_topics.csv", low_memory=False)


# =============================================================================
# Chart 5 -- what's trending in Quebec (matplotlib, ggplot style)
# =============================================================================
# `trending_now` is a snapshot table (no query targeting -- one call returns
# every currently-trending topic in Quebec) re-collected every 12h, so the
# same topic re-appears across sweeps while it's still trending. Ranking by
# that appearance count -- rather than any single sweep's snapshot -- shows
# what's *persistently* trending over the collection window, and doubles as
# an easy way to see whether election topics break into Quebec's general
# attention at all, next to sports/entertainment chatter.
def simplify_trend_category(categories: str) -> str:
    cats = {c.strip() for c in str(categories).split(";")}
    if cats & {"politics", "law_and_government"}:
        return "Politics & government"
    if "sports" in cats:
        return "Sports"
    if "entertainment" in cats:
        return "Entertainment"
    return "Other"


TREND_CATEGORY_COLOR = {
    "Politics & government": "#2a78d6",
    "Sports": "#eb6834",
    "Entertainment": "#4a3aa7",
    "Other": "#b6b5b0",
}

trending_agg = (
    trends_trending_now.groupby("query")
    .agg(n_appearances=("query", "size"),
         categories=("categories", lambda s: s.value_counts().idxmax()))
    .reset_index()
)
trending_agg["category"] = trending_agg["categories"].map(simplify_trend_category)
top_trending = trending_agg.sort_values("n_appearances", ascending=False).head(20).sort_values("n_appearances")

fig_trending, ax_trending = plt.subplots(figsize=(8, 7))
ax_trending.barh(
    top_trending["query"],
    top_trending["n_appearances"],
    color=top_trending["category"].map(TREND_CATEGORY_COLOR),
)
ax_trending.set(
    title="What's trending in Quebec",
    xlabel="sweep appearances (12-hourly Trends scrapes, since collection began)",
)
ax_trending.tick_params(axis="y", labelsize=8)
ax_trending.legend(
    handles=[Patch(color=color, label=cat) for cat, color in TREND_CATEGORY_COLOR.items()],
    loc="lower right", fontsize=8, frameon=False,
)
plt.tight_layout()


# =============================================================================
# Chart 6 -- Google Trends interest for the 5 headline parties, past 12
# months (plotly line chart)
# =============================================================================
# The weekly 12m-window call re-bases its whole series on every run, but the
# two calls collected so far (a week apart) agree almost exactly on their
# overlapping weeks (+-1 index point) -- averaging duplicate (party, week)
# points smooths that negligible re-basing noise rather than picking one
# call arbitrarily. `query` here is Trends' own short code, hand-picked per
# collecter/readme.md ("PCQ" means Parti *conservateur* du Québec here --
# the media acronym -- not the unrelated micro-party that also happens to
# hold the `PCQ` party_key elsewhere in this dataset).
TRENDS_QUERY_TO_PARTY_KEY = {"CAQ": "ÉCF-CAQ", "PQ": "PQ", "PLQ": "PLQ/QLP", "PCQ": "PCOQ", "QS": "QS"}

party_interest = trends_timeseries[
    (trends_timeseries["scope"] == "compare")
    & (trends_timeseries["series"] == "party")
    & (trends_timeseries["window_key"] == "12m")
].copy()
party_interest["party_key"] = party_interest["query"].map(TRENDS_QUERY_TO_PARTY_KEY)
party_interest["week"] = pd.to_datetime(party_interest["timestamp"], unit="s", utc=True)
party_interest = (
    party_interest.groupby(["party_key", "week"])["extracted_value"].mean().reset_index()
)

fig_trends_interest = go.Figure()
for party_key, color in IDENTITY_COLORS.items():
    series = party_interest[party_interest["party_key"] == party_key].sort_values("week")
    fig_trends_interest.add_trace(go.Scatter(
        x=series["week"], y=series["extracted_value"],
        mode="lines+markers", name=PARTY_LABEL[party_key],
        line=dict(color=color, width=2.5), marker=dict(size=5),
        hovertemplate="%{fullData.name}<br>week of %{x|%b %d}: %{y}<extra></extra>",
    ))
fig_trends_interest.update_layout(
    title="Google Trends interest for the 5 headline parties (past 12 months, weekly)",
    xaxis_title="week",
    yaxis_title="relative search interest (Trends-indexed, 0-100)",
    margin=dict(l=0, r=0, t=40, b=0),
)


# =============================================================================
# Chart 7 -- related topics shared by all 5 headline parties (plotly
# node-link diagram)
# =============================================================================
# Using `related_topics`, not `related_queries` -- DECISIONS.md flags
# related_queries' ~53% structural timeout rate (100% for CAQ, 95% for PQ),
# which would silently drop 2-3 of the 5 parties from an "all party"
# comparison. related_topics doesn't share that failure pattern (every
# party has 600+ rows here) and answers the same question: what Google
# associates with searches about each party. `rank_type == "top"` (the
# stable associations, not "rising" breakout noise) over the 12m window,
# same as Chart 6. A topic that shows up in every one of the 5 parties' own
# top-topic lists is as close to a data-driven "common link" as this table
# gets -- no arbitrary top-N cutoff needed.
topic_scores = (
    trends_related_topics[
        (trends_related_topics["rank_type"] == "top") & (trends_related_topics["window_key"] == "12m")
    ]
    # "top" rows are never the "Breakout" sentinel, but `value` is still
    # string-typed at the whole-column level (that sentinel lives in the same
    # column on "rising" rows) -- use the typed `extracted_value` instead.
    .groupby(["scope", "text"])["extracted_value"].max()  # collapse duplicate variants (party-name vs leader-name query)
    .reset_index()
    .rename(columns={"extracted_value": "value"})
)
topics_in_all_5 = topic_scores.groupby("text")["scope"].nunique()
topic_edges = topic_scores[topic_scores["text"].isin(topics_in_all_5[topics_in_all_5 == 5].index)]

party_order = list(IDENTITY_COLORS)
topic_order = topic_edges.groupby("text")["value"].sum().sort_values(ascending=False).index.tolist()
n_parties, n_topics = len(party_order), len(topic_order)

party_y = {party_key: n_parties - 1 - i for i, party_key in enumerate(party_order)}
# Stretch the (longer) topic list to span the same vertical range as the
# party list, so edges fan out cleanly instead of crowding one corner.
topic_y = {
    text: (n_topics - 1 - i) * (n_parties - 1) / max(n_topics - 1, 1)
    for i, text in enumerate(topic_order)
}

fig_related_topics = go.Figure()
for party_key, color in IDENTITY_COLORS.items():
    party_edges = topic_edges[topic_edges["scope"] == party_key].sort_values("value")
    for i, row in enumerate(party_edges.itertuples()):
        fig_related_topics.add_trace(go.Scatter(
            x=[0, 1], y=[party_y[party_key], topic_y[row.text]],
            mode="lines", line=dict(color=color, width=1 + 4 * (row.value / 100) ** 0.5),
            opacity=0.7, name=PARTY_LABEL[party_key], legendgroup=party_key,
            showlegend=(i == len(party_edges) - 1),  # label on the strongest (widest) edge
            hovertext=f"{PARTY_LABEL[party_key]} -> {row.text} (relevance {row.value})",
            hoverinfo="text",
        ))

fig_related_topics.add_trace(go.Scatter(
    x=[0] * n_parties, y=[party_y[p] for p in party_order],
    mode="markers+text", text=[PARTY_LABEL[p] for p in party_order],
    textposition="middle left", textfont=dict(size=12),
    marker=dict(size=12, color=[IDENTITY_COLORS[p] for p in party_order]),
    hoverinfo="skip", showlegend=False,
))
fig_related_topics.add_trace(go.Scatter(
    x=[1] * n_topics, y=[topic_y[t] for t in topic_order],
    mode="markers+text", text=topic_order,
    textposition="middle right", textfont=dict(size=10),
    marker=dict(size=7, color="#52514e"),
    hoverinfo="skip", showlegend=False,
))
fig_related_topics.update_layout(
    title="Related topics shared by all 5 headline parties (Google Trends, 12-month window)",
    xaxis=dict(visible=False, range=[-0.9, 2.4]),
    yaxis=dict(visible=False),
    margin=dict(l=10, r=220, t=40, b=10),
    height=900,
)


# =============================================================================
# Load ChatGPT data -- one row per (party, issue, lang) call, repeated daily
# since collection began (`scheduled_positions`, collecter/app.py).
# =============================================================================
chatgpt = pd.read_csv(SEARCHAPI_DIR / "chatgpt.csv", low_memory=False)
chatgpt["day"] = pd.to_datetime(chatgpt["created_at"], utc=True, errors="coerce").dt.date
FIRST_DAY, LAST_DAY = chatgpt["day"].min(), chatgpt["day"].max()

# The exact prompt template from collecter/app.py's POSITION_PROMPTS -- the
# raw `chatgpt.csv` only carries the party/issue/lang context, not the
# assembled sentence, so it's reconstructed here rather than stored anywhere.
POSITION_PROMPTS = {
    "fr": "Quelle est la position de {party} sur {issue}?",
    "en": "What is {party}'s position on {issue}?",
}

# Shared per-issue bilingual label, used as the dropdown option text for
# both the questions table and the similarity chart below.
issue_label_rows = chatgpt[["issue_key", "issue", "lang"]].drop_duplicates(subset=["issue_key", "lang"])
issue_label_fr = issue_label_rows[issue_label_rows["lang"] == "fr"].set_index("issue_key")["issue"]
issue_label_en = issue_label_rows[issue_label_rows["lang"] == "en"].set_index("issue_key")["issue"]
ISSUE_KEYS = sorted(chatgpt["issue_key"].unique())


def issue_label(issue_key: str) -> str:
    en, fr = issue_label_en[issue_key], issue_label_fr[issue_key]
    return f"{en[:1].upper()}{en[1:]} / {fr[:1].upper()}{fr[1:]}"


# =============================================================================
# Table 1 -- the question template, plus the 13 issues it was asked about
# =============================================================================
# Every one of the 208 (party x issue) questions is the same two sentences
# with just `{party}`/`{issue}` swapped in -- listing all 208 (or even just
# 16-per-issue) out added little beyond the template itself. What actually
# varies and is worth showing is the 13 issues themselves, in both
# languages; the 16 party names are already on every other chart on this
# page.
question_template = bf.Text(f"""
> **FR:** *{POSITION_PROMPTS["fr"].format(party="**{party}**", issue="**{issue}**")}*
>
> **EN:** *{POSITION_PROMPTS["en"].format(party="**{party}**", issue="**{issue}**")}*

`{{party}}` is each of the 16 in-election parties; `{{issue}}` is one of the 13 issues below.
""")

issues_table = bf.Table(
    pd.DataFrame({
        "Issue": ISSUE_KEYS,
        "FR": [issue_label_fr[k] for k in ISSUE_KEYS],
        "EN": [issue_label_en[k] for k in ISSUE_KEYS],
    })
)


# =============================================================================
# Chart -- party answer map, by issue: sentence embeddings + PCA fitted on
# the first day, every later day projected through the same fit (plotly 3D)
# =============================================================================
# Each answer -> 384-d embedding (chatgpt_embeddings.py: granite-97m, full
# answer, no truncation). Then, per issue and per language, PCA(3) is fitted
# on the 16 parties' FIRST-day answers and the same fitted model transforms
# every later day's answers, so all days sit in one shared 3D space and each
# party's trace is its day-to-day drift, not a re-solved layout. Why not one
# fit for everything (checked on day-1 data):
#   - one PCA across all 13 issues spends its 3 axes separating issues from
#     each other and keeps only ~11% of the party-to-party variance within
#     an issue; a per-issue fit keeps ~40%.
#   - within an issue, a joint EN+FR fit spends an axis on language (EN/FR
#     offset ~1 std), so each language gets its own fit.
# PCA over UMAP: it's linear, so projecting unseen later-day answers through
# the day-1 fit is exact rather than an approximate neighbour placement.
# ChatGPT regenerates each answer daily, so small day-to-day wiggle is mostly
# rewording; a real change of position shows as a trace that moves and stays.
ANSWER_DAYS = sorted(chatgpt["day"].unique())
ANSWER_DAY_LABELS = [d.strftime("%b %d") for d in ANSWER_DAYS]
FIRST_DAY_LABEL, LAST_DAY_LABEL = ANSWER_DAY_LABELS[0], ANSWER_DAY_LABELS[-1]
DAY_INDEX = {d: i for i, d in enumerate(ANSWER_DAYS)}

answers_daily = (
    chatgpt.drop_duplicates(subset=["party_key", "issue_key", "lang", "day"])
    .sort_values("day")
    .reset_index(drop=True)
)
answer_vectors = embed_answers(answers_daily["answer_markdown"])

# Full answers for the reading panel under the chart (hover tooltips are too
# small for ~550-token answers). html=False escapes any raw HTML in the
# answer text, and markdown-it refuses javascript:/data: links, so the
# rendered HTML is safe to drop into the page.
_render_answer = MarkdownIt("commonmark", {"html": False}).enable("table").render
answer_traces: dict[str, dict] = {}


def project_issue(issue_key: str, lang: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Day-1-fitted PCA coordinates for every day, plus the fit's explained variance."""
    mask = (answers_daily["issue_key"] == issue_key) & (answers_daily["lang"] == lang)
    rows, vectors = answers_daily[mask], answer_vectors[mask.values]
    pca = PCA(n_components=3).fit(vectors[(rows["day"] == FIRST_DAY).values])
    coords = pca.transform(vectors)
    points = rows[["party_key", "day", "answer_markdown"]].assign(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], vec=list(vectors)
    )
    # Distance from the party's own first-day answer, in the full 384-d space
    # so the number doesn't depend on the projection.
    first_day = {pk: v for pk, d, v in zip(rows["party_key"], rows["day"], vectors) if d == FIRST_DAY}
    points["drift"] = [
        1 - first_day[pk] @ v if pk in first_day else np.nan for pk, v in zip(rows["party_key"], vectors)
    ]
    return points, pca.explained_variance_ratio_


# ONE figure with 2 scenes for all 13 issues: each 3D scene is a WebGL
# context and browsers cap those at ~16 per page ("Too many active WebGL
# contexts. Oldest context will be lost."), so 13 separate 2-scene figures
# (26 contexts) rendered blank. Which issue and which parties are shown is
# driven by the control bar above the chart (see answer_panel's script).
fig_answer_map = make_subplots(
    rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=["English", "French"], horizontal_spacing=0.02,
)
axis_titles: dict[str, dict] = {}
for issue_key in ISSUE_KEYS:
    for col, lang in enumerate(["en", "fr"], start=1):
        points, explained = project_issue(issue_key, lang)
        scene = "scene" if col == 1 else "scene2"
        axis_titles[issue_key] = {
            **axis_titles.get(issue_key, {}),
            **{f"{scene}.{axis}axis.title.text": f"PC{k + 1} ({explained[k]:.0%} var.)"
               for k, axis in enumerate("xyz")},
        }
        for party_key, party_points in points.groupby("party_key"):
            party_points = party_points.sort_values("day")
            color = IDENTITY_COLORS.get(party_key, OTHER_COLOR)
            label = PARTY_LABEL[party_key]
            day_idx = party_points["day"].map(DAY_INDEX).tolist()
            drift_labels = party_points["drift"].map(lambda d: "n/a" if pd.isna(d) else f"{d:.3f}").tolist()
            key = f"{issue_key}|{lang}|{party_key}"
            days = [None] * len(ANSWER_DAYS)
            for i, md, drift in zip(day_idx, party_points["answer_markdown"], drift_labels):
                days[i] = [_render_answer(md), drift]
            # Day-by-day cosine distances (full 384-d) between this party's
            # answers, so the page can report "distance from the start of the
            # selected range" for any range the time slider picks.
            vecs = np.stack(party_points["vec"].to_list())
            pair_dist = 1 - vecs @ vecs.T
            dist = [[None] * len(ANSWER_DAYS) for _ in ANSWER_DAYS]
            for a, da in enumerate(day_idx):
                for b, db in enumerate(day_idx):
                    dist[da][db] = round(float(pair_dist[a, b]), 3)
            answer_traces[key] = {
                "party": f"{PARTY_LABEL[party_key]} -- {party_config[party_key]['fr_title']}",
                "issue": issue_label(issue_key),
                "lang": "English" if lang == "en" else "French",
                "days": days,
                "dist": dist,
            }
            last = len(day_idx) - 1
            fig_answer_map.add_trace(go.Scatter3d(
                # Plain lists, not numpy: plotly.py would binary-encode arrays,
                # and the time slider re-slices these from gd.data in the page.
                x=party_points["x"].tolist(), y=party_points["y"].tolist(), z=party_points["z"].tolist(),
                mode="lines+markers+text",
                # Label each party once, at its latest position.
                text=[label if i == last else "" for i in range(len(day_idx))],
                textposition="top center", textfont=dict(size=10, color="#3d3d3a"),
                line=dict(color=color, width=3),
                marker=dict(
                    size=[6 if d == 0 else 7 if i == last else 3 for i, d in enumerate(day_idx)],
                    symbol=["circle" if d == 0 else "diamond" if i == last else "circle"
                            for i, d in enumerate(day_idx)],
                    color=color, line=dict(width=1, color="white"),
                ),
                # [day label, day index, answer_traces key, distance from the
                # range-start answer, range-start day label] -- the last two are
                # rewritten by the time slider; read by the reading panel.
                customdata=[[ANSWER_DAY_LABELS[d], d, key, dl, FIRST_DAY_LABEL]
                            for d, dl in zip(day_idx, drift_labels)],
                hovertemplate=(
                    f"<b>{label}</b> -- %{{customdata[0]}}<br>"
                    "distance from %{customdata[4]} answer: %{customdata[3]}<br>"
                    "<i>full answer below -- click to pin</i><extra></extra>"
                ),
                meta=[issue_key, party_key],  # lets the control bar filter by issue and party
                visible=issue_key == ISSUE_KEYS[0] and party_key in IDENTITY_COLORS,
                showlegend=False,
            ), row=1, col=col)

# Legend for the marker shapes (party identity comes from the labels and the
# party toggles); these carry no meta, so the control bar never hides them.
for name, symbol, size in [("start of the selected days", "circle", 7),
                           ("each day in between", "circle", 4),
                           ("end of the selected days", "diamond", 7)]:
    fig_answer_map.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers", name=name,
        marker=dict(symbol=symbol, size=size, color="#73726c"),
    ), row=1, col=1)

hidden_ticks = dict(showticklabels=False)
fig_answer_map.update_layout(
    **{scene: dict(aspectmode="cube", xaxis=hidden_ticks, yaxis=hidden_ticks, zaxis=hidden_ticks)
       for scene in ("scene", "scene2")},
    **axis_titles[ISSUE_KEYS[0]],
    width=1200, height=720, margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top"),
    meta="answer-map",  # lets the control bar / reading panel script find this figure
)

# Control bar above the chart: issue picker + party toggles (5 main parties
# on by default). Both filter independently, so switching issue keeps the
# party selection -- plotly's own dropdown can't do that, it resets
# visibility wholesale.
_party_toggle_order = list(IDENTITY_COLORS) + sorted(
    set(answers_daily["party_key"]) - set(IDENTITY_COLORS), key=lambda k: PARTY_LABEL[k]
)
answer_map_controls = bf.HTML(
    '<div class="am-controls">'
    '<label class="am-issue">Issue '
    '<select id="am-issue">'
    + "".join(f'<option value="{k}">{html.escape(issue_label(k))}</option>' for k in ISSUE_KEYS)
    + "</select></label>"
    '<div class="am-parties"><span class="am-label">Parties</span>'
    '<button type="button" data-preset="main">5 main</button>'
    '<button type="button" data-preset="all">All</button>'
    '<button type="button" data-preset="none">None</button>'
    + "".join(
        f'<label class="am-chip" title="{html.escape(party_config[k]["fr_title"])}">'
        f'<input type="checkbox" value="{html.escape(k)}"{" checked" if k in IDENTITY_COLORS else ""}>'
        f'<span class="am-dot" style="background:{IDENTITY_COLORS.get(k, OTHER_COLOR)}"></span>'
        f"{html.escape(PARTY_LABEL[k])}</label>"
        for k in _party_toggle_order
    )
    + "</div>"
    # Two range inputs stacked on one track = a two-handle slider; the thumbs
    # take pointer events, the tracks don't, so both handles stay draggable.
    '<div class="am-range"><span class="am-label">Days</span>'
    '<div class="am-slider"><div class="am-track"></div><div class="am-sel" id="am-sel"></div>'
    f'<input type="range" id="am-from" min="0" max="{len(ANSWER_DAYS) - 1}" value="0" step="1" aria-label="First day shown">'
    f'<input type="range" id="am-to" min="0" max="{len(ANSWER_DAYS) - 1}" value="{len(ANSWER_DAYS) - 1}" step="1" '
    'aria-label="Last day shown"></div>'
    f'<span id="am-range-label" class="am-range-label">{FIRST_DAY_LABEL} &rarr; {LAST_DAY_LABEL}</span>'
    f'<span class="am-note">(3D space stays fitted on {FIRST_DAY_LABEL})</span></div>'
    "</div>"
    "<style>"
    ".am-controls { display: flex; flex-direction: column; gap: 8px; margin: 4px 0 6px; font-size: 0.9em; }"
    ".am-issue select { margin-left: 6px; padding: 4px 8px; border: 1px solid #c9c8c3; border-radius: 6px; "
    "background: #fff; font: inherit; max-width: 100%; }"
    ".am-parties { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }"
    ".am-label { color: #52514e; margin-right: 2px; }"
    ".am-parties button { border: 1px solid #c9c8c3; background: #fff; border-radius: 6px; padding: 2px 10px; "
    "cursor: pointer; font: inherit; }"
    ".am-chip { display: inline-flex; align-items: center; gap: 5px; border: 1px solid #e3e2de; border-radius: 999px; "
    "padding: 2px 10px 2px 6px; cursor: pointer; user-select: none; }"
    ".am-chip input { margin: 0; }"
    ".am-chip:has(input:not(:checked)) { color: #9b9a95; }"
    ".am-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }"
    ".am-range { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }"
    ".am-slider { position: relative; flex: 0 1 420px; min-width: 220px; height: 24px; }"
    ".am-track, .am-sel { position: absolute; top: 10px; height: 4px; border-radius: 2px; }"
    ".am-track { left: 0; right: 0; background: #e3e2de; }"
    ".am-sel { background: #2a78d6; }"
    ".am-slider input[type=range] { position: absolute; left: 0; top: 0; width: 100%; height: 24px; margin: 0; "
    "background: none; pointer-events: none; -webkit-appearance: none; appearance: none; }"
    ".am-slider input[type=range]::-webkit-slider-runnable-track { background: none; height: 24px; }"
    ".am-slider input[type=range]::-moz-range-track { background: none; }"
    ".am-slider input[type=range]::-webkit-slider-thumb { pointer-events: auto; -webkit-appearance: none; "
    "width: 16px; height: 16px; margin-top: 4px; border-radius: 50%; background: #fff; border: 2px solid #2a78d6; cursor: pointer; }"
    ".am-slider input[type=range]::-moz-range-thumb { pointer-events: auto; width: 12px; height: 12px; "
    "border-radius: 50%; background: #fff; border: 2px solid #2a78d6; cursor: pointer; }"
    ".am-range-label { font-weight: 600; min-width: 130px; }"
    ".am-note { color: #73726c; font-size: 0.9em; }"
    "</style>"
)

# Reading panel under the chart: hovering a point shows that party's full
# answer for that day next to its first-day answer; clicking pins it so
# moving the mouse down to read doesn't swap the text. Answers for all 16
# days are ~7x the text of the chart itself, so they're shipped gzipped +
# base64 and unzipped in the browser (DecompressionStream).
_answers_blob = base64.b64encode(
    gzip.compress(json.dumps(answer_traces, ensure_ascii=False).encode("utf-8"), compresslevel=9)
).decode("ascii")
_axis_titles_json = json.dumps(axis_titles)
answer_panel = bf.HTML(f"""
<div id="ap" class="ap">
  <div class="ap-head">
    <div>
      <div id="ap-title" class="ap-title">Hover a point in the chart above to read that answer here</div>
      <div id="ap-sub" class="ap-sub">Click a point to pin it while you read.</div>
    </div>
    <button id="ap-unpin" class="ap-unpin" hidden>Unpin</button>
  </div>
  <div class="ap-cols">
    <section id="ap-col-ref" class="ap-col"><h4 id="ap-h-ref">Start of selected days</h4><div id="ap-ref" class="ap-body"></div></section>
    <section id="ap-col-day" class="ap-col"><h4 id="ap-h-day">Hovered day</h4><div id="ap-day" class="ap-body"></div></section>
  </div>
</div>
<style>
  .ap {{ border: 1px solid #e3e2de; border-radius: 8px; padding: 14px 16px; margin: 8px 0 24px; background: #fff; }}
  .ap-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }}
  .ap-title {{ font-weight: 600; font-size: 1.05em; }}
  .ap-sub {{ color: #73726c; font-size: 0.85em; margin-top: 2px; }}
  .ap-unpin {{ border: 1px solid #c9c8c3; background: #fff; border-radius: 6px; padding: 4px 12px; cursor: pointer; }}
  .ap-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 900px) {{ .ap-cols {{ grid-template-columns: 1fr; }} }}
  .ap-col {{ border-top: 3px solid #e3e2de; padding-top: 6px; min-width: 0; }}
  .ap-col.ap-active {{ border-top-color: #2a78d6; }}
  .ap-col h4 {{ margin: 0 0 6px; font-size: 0.9em; color: #52514e; }}
  .ap-body {{ max-height: 460px; overflow-y: auto; font-size: 0.9em; line-height: 1.5; padding-right: 6px; }}
  /* briefing's global reset zeroes margin/padding on everything, which
     clips list bullets and collapses paragraphs -- restore the same rules
     briefing uses for its own .bf-text blocks. */
  .ap-body p {{ margin-bottom: 0.6rem; }}
  .ap-body ul, .ap-body ol {{ padding-left: 1.4rem; margin-bottom: 0.6rem; }}
  .ap-body li {{ margin-bottom: 0.2rem; }}
  .ap-body h1, .ap-body h2, .ap-body h3, .ap-body h4 {{ font-size: 1em; font-weight: 600; margin: 0.8rem 0 0.3rem; }}
  .ap-body table {{ border-collapse: collapse; margin: 6px 0; }}
  .ap-body th, .ap-body td {{ border: 1px solid #e3e2de; padding: 3px 6px; }}
  .ap-empty {{ color: #73726c; font-style: italic; }}
</style>
<script>
(function () {{
  const AXIS_TITLES = {_axis_titles_json};
  const MAIN_PARTIES = {json.dumps(list(IDENTITY_COLORS))};
  const DAY_LABELS = {json.dumps(ANSWER_DAY_LABELS)};
  const $ = (id) => document.getElementById(id);
  const EMPTY = '<p class="ap-empty">No answer collected that day.</p>';
  let ANSWERS = null, pinned = false;

  const answersReady = (async () => {{
    const bytes = Uint8Array.from(atob("{_answers_blob}"), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    ANSWERS = JSON.parse(await new Response(stream).text());
  }})();

  // Selected day range (indices into DAY_LABELS), from the two-handle slider.
  const rangeLo = () => +$("am-from").value;
  const rangeHi = () => +$("am-to").value;
  // Days a party actually answered within the range (one answer is missing overall).
  const daysInRange = (a) => {{
    const out = [];
    for (let d = rangeLo(); d <= rangeHi(); d++) if (a.days[d]) out.push(d);
    return out;
  }};
  const fmt = (d) => (d == null ? "n/a" : d.toFixed(3));

  function show(key, dayIdx) {{
    const a = ANSWERS && ANSWERS[key];
    if (!a) return;
    const avail = daysInRange(a);
    if (!avail.length) return;
    // Left column: the start of the selected range. Right column: the hovered
    // day -- or, when the start point itself is hovered, the end of the range,
    // so the two columns only show the same answer for a one-day range.
    const left = avail[0];
    const right = dayIdx === left ? avail[avail.length - 1] : dayIdx;
    $("ap-title").textContent = a.party;
    $("ap-sub").textContent = a.issue + " \\u00b7 " + a.lang + " \\u00b7 distance between the " + DAY_LABELS[left]
      + " and " + DAY_LABELS[right] + " answers: " + fmt(a.dist[left][right])
      + (pinned ? " \\u00b7 pinned" : " \\u00b7 click the point to pin");
    $("ap-ref").innerHTML = a.days[left][0];
    $("ap-day").innerHTML = a.days[right] ? a.days[right][0] : EMPTY;
    $("ap-h-ref").textContent = DAY_LABELS[left] + " -- start of selected days";
    $("ap-h-day").textContent = DAY_LABELS[right] + (dayIdx === left ? " -- end of selected days" : " -- hovered day");
    $("ap-col-ref").classList.toggle("ap-active", dayIdx === left);
    $("ap-col-day").classList.toggle("ap-active", dayIdx !== left);
  }}

  function reset() {{
    pinned = false;
    $("ap-unpin").hidden = true;
    $("ap-title").textContent = "Hover a point in the chart above to read that answer here";
    $("ap-sub").textContent = "Click a point to pin it while you read.";
    $("ap-ref").innerHTML = ""; $("ap-day").innerHTML = "";
    $("ap-h-ref").textContent = "Start of selected days"; $("ap-h-day").textContent = "Hovered day";
    $("ap-col-ref").classList.remove("ap-active"); $("ap-col-day").classList.remove("ap-active");
  }}

  function attach() {{
    const gd = [...document.querySelectorAll(".plotly-graph-div")].find((g) => g.layout && g.layout.meta === "answer-map");
    if (!gd || !gd.on) return setTimeout(attach, 200);
    const toggles = [...document.querySelectorAll(".am-chip input")];
    // Full 16-day copy of every party trace, taken before any filtering, so
    // the time range can be re-sliced from it any number of times.
    const FULL = gd.data.map((t) => t.meta && {{
      x: [...t.x], y: [...t.y], z: [...t.z],
      cd: t.customdata.map((c) => [...c]),
      label: t.text.find((s) => s) || "",
    }});
    const traceIdx = FULL.map((f, i) => (f ? i : -1)).filter((i) => i >= 0);

    function sliceTrace(i) {{
      const f = FULL[i], lo = rangeLo(), hi = rangeHi();
      const keep = f.cd.map((c, k) => (c[1] >= lo && c[1] <= hi ? k : -1)).filter((k) => k >= 0);
      const n = keep.length, a = ANSWERS && ANSWERS[f.cd[0][2]];
      const start = n ? f.cd[keep[0]][1] : null;
      return {{
        x: keep.map((k) => f.x[k]), y: keep.map((k) => f.y[k]), z: keep.map((k) => f.z[k]),
        // Distances from the range-start answer (the full day-by-day table
        // ships with the answers; before it has loaded, keep the originals).
        cd: keep.map((k) => {{
          const c = [...f.cd[k]];
          if (a) {{ c[3] = fmt(a.dist[start][c[1]]); c[4] = DAY_LABELS[start]; }}
          return c;
        }}),
        text: keep.map((k, j) => (j === n - 1 ? f.label : "")),
        size: keep.map((k, j) => (j === n - 1 && n > 1 ? 7 : j === 0 ? 6 : 3)),
        symbol: keep.map((k, j) => (j === n - 1 && n > 1 ? "diamond" : "circle")),
      }};
    }}

    function applyFilters(relayout) {{
      const issue = $("am-issue").value;
      const parties = new Set(toggles.filter((t) => t.checked).map((t) => t.value));
      const upd = {{ visible: [], x: [], y: [], z: [], customdata: [], text: [], "marker.size": [], "marker.symbol": [] }};
      for (const i of traceIdx) {{
        const meta = gd.data[i].meta, s = sliceTrace(i);
        upd.visible.push(meta[0] === issue && parties.has(meta[1]));
        upd.x.push(s.x); upd.y.push(s.y); upd.z.push(s.z); upd.customdata.push(s.cd); upd.text.push(s.text);
        upd["marker.size"].push(s.size); upd["marker.symbol"].push(s.symbol);
      }}
      Plotly.update(gd, upd, relayout ? AXIS_TITLES[issue] : {{}}, traceIdx);
    }}

    function syncSlider(moved) {{
      const from = $("am-from"), to = $("am-to"), max = +from.max;
      if (+from.value > +to.value) (moved === from ? to : from).value = moved.value;
      // When both handles sit at the right end, lift "from" so it can still be dragged back.
      from.style.zIndex = +from.value === max ? 3 : 1;
      $("am-sel").style.left = (100 * from.value / max) + "%";
      $("am-sel").style.width = (100 * (to.value - from.value) / max) + "%";
      const n = to.value - from.value + 1;
      $("am-range-label").textContent = DAY_LABELS[from.value] + " \\u2192 " + DAY_LABELS[to.value]
        + " (" + n + (n === 1 ? " day)" : " days)");
    }}

    $("am-issue").addEventListener("change", () => {{ reset(); applyFilters(true); }});
    toggles.forEach((t) => t.addEventListener("change", () => applyFilters(false)));
    document.querySelectorAll(".am-parties button").forEach((b) => b.addEventListener("click", () => {{
      toggles.forEach((t) => {{
        t.checked = b.dataset.preset === "all" || (b.dataset.preset === "main" && MAIN_PARTIES.includes(t.value));
      }});
      applyFilters(false);
    }}));
    for (const id of ["am-from", "am-to"]) {{
      $(id).addEventListener("input", (e) => {{ syncSlider(e.target); reset(); applyFilters(false); }});
    }}
    syncSlider($("am-to"));
    answersReady.then(() => applyFilters(false));

    let hovered = null, downAt = null;
    gd.on("plotly_hover", (e) => {{
      hovered = [e.points[0].customdata[2], e.points[0].customdata[1]];
      if (!pinned) show(...hovered);
    }});
    gd.on("plotly_unhover", () => {{ hovered = null; }});
    // plotly doesn't emit plotly_click for scatter3d points reliably, so pin
    // on a native press+release that didn't move (a drag rotates the scene).
    gd.addEventListener("mousedown", (e) => {{ downAt = [e.clientX, e.clientY]; }}, true);
    gd.addEventListener("mouseup", (e) => {{
      const still = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5;
      downAt = null;
      if (still && hovered) {{ pinned = true; $("ap-unpin").hidden = false; show(...hovered); }}
    }}, true);
    $("ap-unpin").addEventListener("click", reset);
  }}
  attach();
}})();
</script>
""")


# =============================================================================
# Sources page -- which sites ChatGPT cites in its answers, day by day
# =============================================================================
# chatgpt_references = the links ChatGPT actually cited in each answer
# (`web_results` is only its raw search hits, not necessarily used). It
# carries no day or call id, but its rows come out in the same order as
# chatgpt.csv's answers, `n_reference_links` rows per answer -- so repeating
# each answer's day that many times lines them up. Asserted on every row
# below, so a future export that reorders either file fails loudly instead
# of silently mis-dating citations.
chatgpt_refs = pd.read_csv(SEARCHAPI_DIR / "chatgpt_references.csv", low_memory=False)
refs_per_answer = chatgpt["n_reference_links"].fillna(0).astype(int)
answer_of_ref = chatgpt.loc[chatgpt.index.repeat(refs_per_answer)].reset_index(drop=True)
assert len(answer_of_ref) == len(chatgpt_refs), "chatgpt_references no longer lines up with chatgpt.csv"
assert (answer_of_ref[["party_key", "issue_key", "lang"]].values
        == chatgpt_refs[["party_key", "issue_key", "lang"]].values).all(), \
    "chatgpt_references rows are out of order vs chatgpt.csv answers"
chatgpt_refs["day"] = answer_of_ref["day"].values

# Count by registrable domain (Public Suffix List, bundled snapshot -- no
# network): the free-text `source` field has ~850 spellings for ~500 sites
# ("Parti Québécois" and "PQ Official Site" are both pq.org), and subdomains
# of one organisation (api-wp.quebecsolidaire.net, candidate.pvq.qc.ca) or
# ministries under gouv.qc.ca belong together.
_domain_of = tldextract.TLDExtract(suffix_list_urls=())
chatgpt_refs["source_domain"] = chatgpt_refs["link"].map(lambda url: _domain_of(url).top_domain_under_public_suffix)

SOURCE_DAYS = sorted(chatgpt_refs["day"].unique())
SOURCE_DAY_LABELS = [d.strftime("%b %d") for d in SOURCE_DAYS]
TOP_SOURCES = 20
# Blue sequential ramp (dataviz palette, steps 100 -> 700); zero cells are
# left blank rather than painted the lightest step, so "never cited that
# day" reads differently from "cited once".
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

source_counts = (
    chatgpt_refs.groupby(["party_key", "source_domain", "day"]).size().rename("n").reset_index()
)


def build_sources_heatmap(counts: pd.DataFrame, title: str) -> go.Figure:
    grid = (
        counts.pivot_table(index="source_domain", columns="day", values="n", aggfunc="sum", fill_value=0)
        .reindex(columns=SOURCE_DAYS, fill_value=0)
    )
    totals = grid.sum(axis=1).sort_values(ascending=False)
    grid = grid.loc[totals.index[:TOP_SOURCES]]
    day_totals = counts.groupby("day")["n"].sum().reindex(SOURCE_DAYS)
    share = grid / day_totals.values
    row_labels = [f"{domain}  ({totals[domain]:,})" for domain in grid.index]

    fig = go.Figure(go.Heatmap(
        z=grid.where(grid > 0).values, x=SOURCE_DAY_LABELS, y=row_labels,
        colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        zmin=0, xgap=2, ygap=2,
        texttemplate="%{z}", textfont=dict(size=10),
        customdata=share.values,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} citations (%{customdata:.1%} of that day's)<extra></extra>",
        colorbar=dict(title="citations", thickness=12),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(side="top", tickangle=0),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor="white",
        width=1200, height=90 + 30 * len(grid), margin=dict(l=10, r=10, t=90, b=10),
    )
    return fig


def party_option_label(party_key: str) -> str:
    return f"{PARTY_LABEL[party_key]} -- {party_config[party_key]['fr_title']}"


source_party_order = list(IDENTITY_COLORS) + sorted(
    (k for k in source_counts["party_key"].unique() if k not in IDENTITY_COLORS), key=lambda k: PARTY_LABEL[k]
)
sources_select = bf.Select(
    bf.Plot(
        build_sources_heatmap(source_counts, f"Top {TOP_SOURCES} sources cited by ChatGPT, all 16 parties"),
        label="All parties",
    ),
    *[
        bf.Plot(
            build_sources_heatmap(
                source_counts[source_counts["party_key"] == party_key],
                f"Top sources cited by ChatGPT when asked about {party_option_label(party_key)}",
            ),
            label=party_option_label(party_key),
        )
        for party_key in source_party_order
    ],
    type=bf.SelectType.DROPDOWN,
)

# Full numbers, every source (not just each view's top 20), searchable.
sources_table = (
    source_counts.assign(day=lambda d: pd.to_datetime(d["day"]).dt.strftime("%b %d"))
    .pivot_table(index=["party_key", "source_domain"], columns="day", values="n", aggfunc="sum", fill_value=0)
    .reindex(columns=SOURCE_DAY_LABELS, fill_value=0)
)
sources_table.insert(0, "Total", sources_table.sum(axis=1))
sources_table = (
    sources_table.reset_index()
    .assign(Party=lambda d: d["party_key"].map(PARTY_LABEL))
    .rename(columns={"source_domain": "Source"})
    .sort_values(["Party", "Total"], ascending=[True, False])
    [["Party", "Source", "Total", *SOURCE_DAY_LABELS]]
    .reset_index(drop=True)
)


# =============================================================================
# Footer -- credit line for the briefing package
# =============================================================================
FOOTER = bf.HTML(
    '<p style="text-align:center;color:#888;font-size:0.85em;margin-top:2em;">'
    'Built with <a href="https://pypi.org/project/briefing/" target="_blank" rel="noopener">briefing</a> 💖'
    "</p>"
)


# =============================================================================
# Assemble + save the report
# =============================================================================
page_dgeq = bf.Page(
    bf.Text("# Quebec 2026 election -- DGEQ overview"),
    bf.Plot(fig_candidates, caption="Number of candidates per party"),
    bf.Plot(fig_electors, caption="Electors per km2 by circonscription"),
    FOOTER,
    title="DGEQ",
)

page_search = bf.Page(
    bf.Text("# Quebec 2026 election -- Google News & Google Rank"),
    bf.Plot(fig_news, caption="Daily unique Google News articles per party, since collection began"),
    bf.Plot(fig_rank, caption="Party leaders and their #1-ranked Google search result"),
    FOOTER,
    title="Search & News",
)

page_trends = bf.Page(
    bf.Text("# Quebec 2026 election -- Google Trends"),
    bf.Plot(fig_trending, caption="What's trending in Quebec"),
    bf.Plot(fig_trends_interest, caption="Google Trends interest for the 5 headline parties"),
    bf.Plot(fig_related_topics, caption="Related topics shared by all 5 headline parties"),
    FOOTER,
    title="Trends",
)

page_chatgpt = bf.Page(
    bf.Text("# Quebec 2026 election -- ChatGPT"),
    bf.Text("## The question template"),
    question_template,
    issues_table,
    bf.Text(
        "## Party answer map by issue\n"
        "Each ChatGPT answer is embedded with a multilingual sentence-embedding model "
        "(`granite-embedding-97m-multilingual-r2`, whole answer, no truncation), then projected to 3D "
        "with a PCA **fitted on the first day's answers** for that issue and language. **Every later day's "
        "answers are projected with that same fitted PCA**, so all days share one space: closer points "
        "mean more similar answers, and each trace follows one party's answer day by day. ChatGPT rewrites "
        "its answer every day, so small wiggles are mostly rewording; a trace that moves away and stays "
        "there is a real change. The 5 main parties are shown by default -- toggle any party below. Drag to "
        "rotate; hover a point to read that day's full answer next to the first day's, in the panel below."
    ),
    answer_map_controls,
    bf.Plot(fig_answer_map, caption="Each trace: one party's answer day by day -- circle: first day, diamond: latest day"),
    answer_panel,
    FOOTER,
    title="ChatGPT",
)

page_sources = bf.Page(
    bf.Text("# Quebec 2026 election -- ChatGPT sources"),
    bf.Text(
        "Which websites does ChatGPT cite when answering our 208 party x issue questions? One "
        "**citation** = one link ChatGPT cited in one answer (English and French answers combined, "
        f"~415 answers and ~1,700 citations per day). Sources are grouped by site domain. Pick "
        "*All parties* or a single party; each view shows its top 20 sources, with each source's "
        "total next to its name. Blank cells: not cited that day."
    ),
    sources_select,
    bf.Text("## Every source, party by party\nSearch by party or site; one column per day."),
    bf.DataTable(sources_table, caption="Citations per source per day, by party"),
    FOOTER,
    title="ChatGPT sources",
)

report = bf.Briefing(page_dgeq, page_search, page_trends, page_chatgpt, page_sources)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
bf.save(report, str(OUT_PATH), name="Quebec 2026 Election")
print(f"Report written to {OUT_PATH}")

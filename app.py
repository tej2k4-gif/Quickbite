# app.py — QuickBite Diner Intelligence (interactive edition)
# ------------------------------------------------------------
# Run locally:   streamlit run app.py
# Needs in the same folder:
#   food_models.joblib   created by QuickBite_Analysis.ipynb (same format as the brief)
#   diners.csv, orders.csv   used by the exploration pages
# ------------------------------------------------------------
import html
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from packaging.version import Version
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="QuickBite Diner Intelligence", page_icon="🍔", layout="wide")
NEW_ST = Version(st.__version__) >= Version("1.50")
SEED = 42

# ------------------------------------------------------------ look and feel
BASIL, INK, BG, PANEL, MUTED, GRID, CHILI = "#2F6B4F", "#1E2A2F", "#F6F8F7", "#E9EFEB", "#5F6E69", "#DCE4DF", "#C0463B"
KNOWN_SEG_COLORS = {"Budget Students": "#D9962B", "Family Feasters": "#C0463B", "Premium Foodies": "#5A4A86"}
EXTRA_COLORS = ["#2F6B4F", "#3E7CB1", "#8C6D3F", "#B5577E", "#4F9DA6", "#7A8B3A", "#9A6B2F", "#556270"]
FONT = "'Source Sans Pro', 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif"
HEAD_FONT = "'Bricolage Grotesque', " + FONT

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;600&display=swap');
h1, h2, h3, h4, h5 {{ font-family: {HEAD_FONT} !important; letter-spacing: -0.01em; color: {INK}; }}
h1 {{ font-weight: 700 !important; }}
.hero {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: clamp(2rem, 4.2vw, 3.2rem);
        line-height: 1.04; letter-spacing: -0.025em; color: {INK}; margin: 0.2rem 0 0.6rem; }}
.lede {{ max-width: 68ch; color: {MUTED}; font-size: 1.06rem; line-height: 1.5; margin-bottom: 1.1rem; }}
.splitbar {{ display: flex; height: 48px; border-radius: 6px; overflow: hidden; margin: 0.4rem 0 1.4rem; }}
.splitbar > div {{ display: flex; align-items: center; padding-left: 14px; color: #fff; font-weight: 600;
                  white-space: nowrap; overflow: hidden; font-size: 0.98rem; }}
.seg-name {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.35rem; line-height: 1.15; margin-bottom: 0.3rem; }}
.ticket {{ background: #fff; color: {INK}; font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 0.9rem; padding: 24px 24px 26px; position: relative; margin: 8px 0 20px;
          box-shadow: 0 1px 2px rgba(30,42,47,.10), 0 6px 18px rgba(30,42,47,.06); }}
.ticket::before, .ticket::after {{ content: ""; position: absolute; left: 0; right: 0; height: 10px;
          background-image: radial-gradient(circle at 7px 5px, {BG} 4.6px, transparent 5.1px);
          background-size: 14px 10px; background-repeat: repeat-x; }}
.ticket::before {{ top: -5px; }}
.ticket::after {{ bottom: -5px; }}
.ticket hr {{ border: none; border-top: 1.5px dashed #AEB9B4; margin: 13px 0; }}
.t-head {{ display: flex; justify-content: space-between; font-weight: 600; }}
.t-small {{ color: {MUTED}; font-size: 0.78rem; margin-top: 3px; }}
.t-label {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 4px; }}
.t-seg {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.9rem; line-height: 1.1; }}
.t-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 2px 0; }}
.t-stamp {{ display: inline-block; margin-top: 10px; border: 2.5px solid; border-radius: 3px; padding: 3px 12px;
           font-weight: 600; font-size: 1.02rem; transform: rotate(-3deg); }}
</style>
""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def nice(item):
    return str(item).replace("_", " ")


def show_chart(fig):
    if NEW_ST:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    if NEW_ST:
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)


def style(fig, h=360, legend=True):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=30 if legend else 12, b=8),
                      font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
                      hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def learn(text):
    if st.session_state.get("learn", True):
        st.info(text, icon="🎓")


# ------------------------------------------------------------ models and data
if not os.path.exists("food_models.joblib"):
    st.error("food_models.joblib is missing. Run QuickBite_Analysis.ipynb in this folder to create it, then refresh.")
    st.stop()


@st.cache_resource
def load_models():
    return joblib.load("food_models.joblib")


M = load_models()
scaler, kmeans, clf, FEATURES, reco = M["scaler"], M["kmeans"], M["clf"], list(M["features"]), M["reco"]
seg_names = {int(k): v for k, v in M["segment_names"].items()}
SEG_ORDER = [seg_names[i] for i in sorted(seg_names)]
CLF_FEATURES = FEATURES + ["age", "total_spend"]


def seg_color(name):
    if name in KNOWN_SEG_COLORS:
        return KNOWN_SEG_COLORS[name]
    others = [s for s in SEG_ORDER if s not in KNOWN_SEG_COLORS]
    idx = others.index(name) if name in others else abs(hash(name))
    return EXTRA_COLORS[idx % len(EXTRA_COLORS)]


LABELS = {"orders_per_month": "Orders per month", "avg_order_value": "Avg order value (AED)",
          "avg_items_per_order": "Items per order", "days_since_last_order": "Days since last order",
          "discount_rate": "Discount usage", "age": "Age", "total_spend": "Total spend (AED)"}
lab = lambda f: LABELS.get(f, f)
NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


@st.cache_data
def load_data():
    if not (os.path.exists("diners.csv") and os.path.exists("orders.csv")):
        return None, None
    return pd.read_csv("diners.csv"), pd.read_csv("orders.csv")


diners_raw, orders = load_data()
HAS_DATA = diners_raw is not None


@st.cache_data
def prepare_diners(d):
    d = d.copy()
    d["cluster"] = kmeans.predict(scaler.transform(d[FEATURES]))
    d["segment"] = d["cluster"].map(seg_names)
    d["monthly_value"] = d["orders_per_month"] * d["avg_order_value"]
    return d


if HAS_DATA:
    diners = prepare_diners(diners_raw)
    orders_seg = orders.merge(diners[["diner_id", "segment"]], on="diner_id", how="left")
    X_SCALED = scaler.transform(diners[FEATURES])


@st.cache_data
def basket_matrix(scope):
    lines = orders_seg if scope == "All baskets" else orders_seg[orders_seg["segment"] == scope]
    baskets = lines.groupby("order_id")["item"].apply(list).tolist()
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)


RULE_COLS = ["if_bought", "then_also", "support", "confidence", "lift", "baskets", "antecedents", "consequents", "itemset"]


@st.cache_data
def mine_rules(scope, min_support):
    X = basket_matrix(scope)
    freq = apriori(X, min_support=min_support, use_colnames=True)
    if freq.empty or freq["itemsets"].apply(len).max() < 2:
        return pd.DataFrame(columns=RULE_COLS)
    try:
        r = association_rules(freq, num_itemsets=len(X), metric="lift", min_threshold=0.0)
    except TypeError:
        r = association_rules(freq, metric="lift", min_threshold=0.0)
    r = r[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    r["antecedents"] = r["antecedents"].apply(lambda s: tuple(sorted(s)))
    r["consequents"] = r["consequents"].apply(lambda s: tuple(sorted(s)))
    r["if_bought"] = r["antecedents"].apply(lambda t: " + ".join(map(nice, t)))
    r["then_also"] = r["consequents"].apply(lambda t: " + ".join(map(nice, t)))
    r["itemset"] = r.apply(lambda x: " + ".join(sorted(x["antecedents"] + x["consequents"])), axis=1)
    r["lift"] = r["lift"].round(3)
    r["baskets"] = (r["support"] * len(X)).round().astype(int)
    return r[RULE_COLS].reset_index(drop=True)


def best_rules(rules, min_conf=0.5, min_lift=1.0, dedup=True, sort="lift"):
    r = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)]
    r = r.sort_values(["lift", "confidence", "support"], ascending=False)
    if dedup:
        r = r.drop_duplicates(subset="itemset", keep="first")
    return r.sort_values([sort, "confidence"], ascending=False).reset_index(drop=True)


def item_shares(scope):
    return basket_matrix(scope).mean().sort_values(ascending=False)


def top_items(seg, n=3):
    items = item_shares(seg).head(n).index.tolist() if HAS_DATA else reco.get(seg, [])[:n]
    return [nice(i) for i in items]


WORDS = {"orders_per_month": ("orders often", "orders rarely"),
         "avg_order_value": ("spends a lot per order", "spends little per order"),
         "avg_items_per_order": ("big baskets", "small baskets"),
         "days_since_last_order": ("hasn't ordered lately", "ordered recently"),
         "discount_rate": ("leans on discounts", "rarely uses discounts")}


def describe_z(zrow):
    top = zrow.abs().sort_values(ascending=False).index[:3]
    parts = [WORDS[f][0] if zrow[f] > 0 else WORDS[f][1] for f in top if abs(zrow[f]) > 0.3 and f in WORDS]
    text = ", ".join(parts) if parts else "close to the average diner"
    if 0 < len(parts) < 2:
        text += "; otherwise close to the average diner"
    return text[0].upper() + text[1:]


@st.cache_data
def segment_z():
    return pd.DataFrame(X_SCALED, columns=FEATURES).assign(segment=diners["segment"].values).groupby("segment").mean()


# ------------------------------------------------------------ classifier lab helpers
SAVED_MODEL = "Saved model (food_models.joblib)"
MODEL_NAMES = [SAVED_MODEL, "Logistic regression", "Decision tree", "Random forest", "Gradient boosting"]


def make_model(name):
    if name == SAVED_MODEL:
        return clone(clf)
    return {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
            "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED),
            "Gradient boosting": GradientBoostingClassifier(random_state=SEED)}[name]


@st.cache_resource
def train(name, test_size):
    X, y = diners[CLF_FEATURES], diners["reordered"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    model = make_model(name).fit(Xtr, ytr)
    return model, Xte, yte.to_numpy(), model.predict_proba(Xte)[:, 1]


def confusion_at(y, proba, t):
    pred = proba >= t
    return dict(caught=int(((y == 0) & ~pred).sum()), missed=int(((y == 0) & pred).sum()),
                false_alarm=int(((y == 1) & ~pred).sum()), kept=int(((y == 1) & pred).sum()))


# ------------------------------------------------------------ session state
PRESETS = {
    "Custom": None,
    "Uni student who orders most days": dict(age=21, orders_per_month=15, avg_order_value=40, avg_items_per_order=2,
                                             days_since_last_order=3, discount_rate=0.70, total_spend=600),
    "Student starting to drift": dict(age=23, orders_per_month=6, avg_order_value=45, avg_items_per_order=2,
                                      days_since_last_order=16, discount_rate=0.60, total_spend=400),
    "Parent planning weekend feasts": dict(age=42, orders_per_month=3, avg_order_value=180, avg_items_per_order=6,
                                           days_since_last_order=20, discount_rate=0.30, total_spend=550),
    "Family that still orders weekly": dict(age=38, orders_per_month=6, avg_order_value=140, avg_items_per_order=5,
                                            days_since_last_order=6, discount_rate=0.45, total_spend=900),
    "Sushi-loving professional": dict(age=31, orders_per_month=10, avg_order_value=95, avg_items_per_order=3,
                                      days_since_last_order=6, discount_rate=0.05, total_spend=950),
    "Professional who's gone quiet": dict(age=34, orders_per_month=8, avg_order_value=110, avg_items_per_order=3,
                                          days_since_last_order=21, discount_rate=0.05, total_spend=880),
}
P_KEYS = {"age": "p_age", "orders_per_month": "p_opm", "avg_order_value": "p_aov", "avg_items_per_order": "p_items",
          "days_since_last_order": "p_rec", "discount_rate": "p_disc", "total_spend": "p_total"}
SLIDER_RANGES = {"age": (18, 70), "orders_per_month": (1, 20), "avg_order_value": (15, 250), "avg_items_per_order": (1, 10),
                 "days_since_last_order": (0, 60), "discount_rate": (0.0, 1.0), "total_spend": (50, 3000)}
PAGES = ["🏠 Overview", "🔮 Diner predictor", "🧩 Segments", "🛒 Basket rules", "🔁 Reorder model", "🚨 At-risk list", "💡 Takeaways"]

DEFAULTS = dict(page=PAGES[0], learn=True, threshold=0.50, preset="Custom",
                p_age=28, p_opm=8, p_aov=90, p_items=3, p_rec=10, p_disc=0.30, p_total=800,
                pv_view="2-D map", pv_x="days_since_last_order", pv_y="avg_order_value", wf_feat="days_since_last_order",
                seg_explore=False, seg_k=3, seg_view="2-D map", seg_x="orders_per_month", seg_y="avg_order_value",
                seg_z="discount_rate", seg_centroids=True, seg_units="Real units", seg_chart="Heatmap",
                seg_dist="avg_order_value", seg_points=False, seg_filter=[],
                b_scope="All baskets", b_sup=0.02, b_conf=0.50, b_lift=1.0, b_dedup=True, b_sort="lift",
                b_topn=15, b_items=[], b_pair="Lift", b_explain=None,
                m_model=SAVED_MODEL, m_test=0.25, m_fa=20, m_miss=200, m_view="Probability split", m_compare=False,
                r_segs=[], r_only=True, r_range=(0.0, 1.0), r_sort="Lowest reorder chance")
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
    else:  # keep widget values when you switch pages
        st.session_state[k] = st.session_state[k]


def apply_preset():
    p = PRESETS.get(st.session_state.preset)
    if p:
        for f, key in P_KEYS.items():
            st.session_state[key] = p[f]


def mark_custom():
    st.session_state.preset = "Custom"


def go_to(page):
    st.session_state.page = page


def set_threshold(t):
    st.session_state.threshold = float(t)


def reset_support():
    st.session_state.b_sup = 0.02 if st.session_state.b_scope == "All baskets" else 0.05
    st.session_state.b_items = []
    st.session_state.b_explain = None


def current_profile():
    return {f: st.session_state[k] for f, k in P_KEYS.items()}


def analyse(profile, threshold):
    """Same logic as the original app.py, plus distances and a custom decision threshold."""
    row = pd.DataFrame([profile])
    z = scaler.transform(row[FEATURES])
    seg = int(kmeans.predict(z)[0])
    name = seg_names.get(seg, f"Segment {seg}")
    prob = float(clf.predict_proba(row[CLF_FEATURES])[0][1])
    return dict(seg=seg, name=name, prob=prob, reorder=int(prob >= threshold), items=reco.get(name, []),
                z=z[0], dists=np.linalg.norm(kmeans.cluster_centers_ - z, axis=1))


def batch_predict(rows):
    df = pd.DataFrame(rows)
    segs = kmeans.predict(scaler.transform(df[FEATURES]))
    return clf.predict_proba(df[CLF_FEATURES])[:, 1], [seg_names.get(int(s), f"Segment {s}") for s in segs]


if not HAS_DATA and st.session_state.page != "🔮 Diner predictor":
    st.session_state.page = "🔮 Diner predictor"
    st.warning("diners.csv and orders.csv aren't in this folder, so only the Diner predictor page can run. "
               "Copy both files next to app.py to unlock the other pages.")

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🍔 QuickBite")
    st.caption("Diner intelligence dashboard")
    st.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.toggle("🎓 Learn mode", key="learn", help="Show short explanations next to each chart and method.")



# ============================================================ pages
def page_overview():
    n, k = len(diners), len(SEG_ORDER)
    counts = diners["segment"].value_counts().reindex(SEG_ORDER)
    st.markdown(f'<div class="hero">{n:,} diners, {NUM_WORDS.get(k, k)} kinds of appetite.</div>', unsafe_allow_html=True)
    st.markdown('<p class="lede">QuickBite asked three questions: who are our diners, what do they order together, '
                'and who is about to stop ordering? Each page answers one of them using the models trained in the notebook.</p>',
                unsafe_allow_html=True)
    bar = "".join(f'<div style="width:{c / n * 100:.1f}%;background:{seg_color(s)}" title="{esc(s)}: {c} diners">'
                  f'{esc(s)} {c / n:.0%}</div>' for s, c in counts.items())
    st.markdown(f'<div class="splitbar">{bar}</div>', unsafe_allow_html=True)

    zs = segment_z()
    for col, seg in zip(st.columns(k), SEG_ORDER):
        d = diners[diners["segment"] == seg]
        with col:
            st.markdown(f'<div class="seg-name" style="color:{seg_color(seg)}">{esc(seg)}</div>', unsafe_allow_html=True)
            st.write(f"{describe_z(zs.loc[seg])}.")
            st.markdown(f"**{d['reordered'].mean():.0%}** reorder  \n"
                        f"AED {d['avg_order_value'].mean():.0f} per order, {d['orders_per_month'].mean():.0f} orders a month  \n"
                        f"Usually orders {', '.join(top_items(seg, 3))}")

    st.divider()
    c = st.columns(4)
    c[0].metric("Diners", f"{n:,}")
    c[1].metric("Orders analysed", f"{orders['order_id'].nunique():,}")
    c[2].metric("Reorder rate", f"{diners['reordered'].mean():.0%}")
    c[3].metric("Average order value", f"AED {diners['avg_order_value'].mean():.0f}")

    st.subheader("Where to start")
    nav = [("🔮 Diner predictor", "Score one diner", "Type in a profile and get their segment, reorder chance and what to recommend."),
           ("🧩 Segments", "See the segments", "Why there are three groups, and how they differ."),
           ("🛒 Basket rules", "Find bundles", "Which items sell together, overall and per segment."),
           ("🔁 Reorder model", "Test the reorder model", "Move the decision line and watch the errors and their cost change."),
           ("🚨 At-risk list", "Get the at-risk list", "Every diner the model flags, with win-back items. Downloadable."),
           ("💡 Takeaways", "Read the takeaway", "Which segment to focus on and which bundle to promote.")]
    cols = st.columns(2)
    for i, (page, btn, desc) in enumerate(nav):
        with cols[i % 2]:
            st.button(btn, key=f"nav_{i}", on_click=go_to, args=(page,))
            st.caption(desc)

    young = diners[diners["age"] < 18]
    with st.expander("Data quality notes"):
        st.write(f"No missing values. {len(young)} diners are listed as under 18 (youngest {young['age'].min()}), "
                 "which looks like an entry error. Age isn't used for segmenting, so the segments aren't affected.")
        show_df(diners.drop(columns=["cluster"]).head(50), hide_index=True)


def page_predictor():
    with st.sidebar:
        st.subheader("Diner profile")
        st.selectbox("Start from a persona", list(PRESETS), key="preset", on_change=apply_preset)
        st.slider("Age", 18, 70, key="p_age", on_change=mark_custom)
        st.slider("Orders per month", 1, 20, key="p_opm", on_change=mark_custom)
        st.slider("Average order value (AED)", 15, 250, key="p_aov", on_change=mark_custom)
        st.slider("Average items per order", 1, 10, key="p_items", on_change=mark_custom)
        st.slider("Days since last order", 0, 60, key="p_rec", on_change=mark_custom)
        st.slider("Discount usage rate", 0.0, 1.0, step=0.05, key="p_disc", on_change=mark_custom)
        st.number_input("Total spend so far (AED)", 50, 20000, step=50, key="p_total", on_change=mark_custom)
        st.subheader("Decision rule")
        st.slider("Reorder threshold", 0.05, 0.95, step=0.01, key="threshold",
                  help="A diner counts as 'will reorder' when their reorder chance is at or above this line. "
                       "The original dashboard uses 0.50.")

    thr = st.session_state.threshold
    profile = current_profile()
    res = analyse(profile, thr)
    name, prob, reorder = res["name"], res["prob"], res["reorder"]

    st.title("Diner predictor")
    st.caption("Change the profile in the sidebar, or start from a persona. Everything updates as you move a slider.")

    if HAS_DATA:
        outside = [lab(f) for f in CLF_FEATURES
                   if not (diners[f].min() <= profile[f] <= diners[f].max())]
        if outside:
            st.warning(f"Outside the range seen in QuickBite's data: {', '.join(outside)}. "
                       "The models are guessing beyond what they learned, so treat this result with care.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        shares = item_shares(name) if HAS_DATA else pd.Series(dtype=float)
        items_html = "".join(f'<div class="t-row"><span>{esc(nice(it))}</span>'
                             f'<span>{shares.get(it, np.nan):.0%} of baskets</span></div>' if it in shares
                             else f'<div class="t-row"><span>{esc(nice(it))}</span></div>' for it in res["items"]) or "<div>No items saved</div>"
        bundle_html = "<div>Load the order data to see bundles</div>"
        if HAS_DATA:
            br = best_rules(mine_rules(name, 0.05))
            if len(br):
                b = br.iloc[0]
                bundle_html = (f'<div>{esc(b["if_bought"])}</div><div>+ {esc(b["then_also"])}</div>'
                               f'<div class="t-small">{b["confidence"]:.0%} of baskets with the first part add the second '
                               f'(lift {b["lift"]:.1f})</div>')
        vcol = BASIL if reorder else CHILI
        ticket = (f'<div class="ticket">'
                  f'<div class="t-head"><span>QuickBite</span><span>Diner ticket</span></div>'
                  f'<div class="t-small">Age {profile["age"]}, {profile["orders_per_month"]} orders a month, '
                  f'AED {profile["avg_order_value"]} per order, last order {profile["days_since_last_order"]} days ago</div><hr>'
                  f'<div class="t-label">Segment</div><div class="t-seg" style="color:{seg_color(name)}">{esc(name)}</div><hr>'
                  f'<div class="t-row"><span>Chance of reordering</span><b>{prob:.0%}</b></div>'
                  f'<div class="t-row"><span>Decision line</span><span>{thr:.0%}</span></div>'
                  f'<div class="t-stamp" style="color:{vcol};border-color:{vcol}">{"Will reorder" if reorder else "At risk"}</div><hr>'
                  f'<div class="t-label">Recommend</div>{items_html}<hr>'
                  f'<div class="t-label">Bundle to suggest</div>{bundle_html}</div>')
        st.markdown(ticket, unsafe_allow_html=True)

    with right:
        st.markdown("##### Chance of reordering")
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number=dict(suffix="%", valueformat=".0f", font=dict(size=48, color=INK)),
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=vcol, thickness=0.32), bgcolor="white",
                       steps=[dict(range=[0, thr * 100], color="#F5E1DE"), dict(range=[thr * 100, 100], color="#DDEDE3")],
                       threshold=dict(line=dict(color=INK, width=3), thickness=0.9, value=thr * 100))))
        g = style(g, h=250, legend=False)
        g.update_layout(margin=dict(l=40, r=40, t=20, b=10))
        show_chart(g)
        st.markdown("##### Next best action")
        st.markdown(next_action(name, reorder))
        learn("The red zone is below the decision line, the green zone above it. Moving the threshold in the sidebar "
              "changes who counts as at risk without retraining anything.")

    tabs = st.tabs(["What drives the score", "Why this segment", "Where they sit", "What-if", "Segment bundles"])
    with tabs[0]:
        tab_drivers(profile, prob)
    with tabs[1]:
        tab_why_segment(res)
    with tabs[2]:
        tab_where(profile, res)
    with tabs[3]:
        tab_whatif(profile, res, thr)
    with tabs[4]:
        tab_bundles(name)


def next_action(name, reorder):
    actions = {
        ("Family Feasters", 0): "Send a family meal deal built on their usual basket (the pizza night or biryani with raita) "
                                "within the next few days, before they drift further.",
        ("Family Feasters", 1): "They're still ordering. Grow the basket: offer the add-on that completes their usual order, "
                                "such as garlic bread with pizza.",
        ("Premium Foodies", 0): "Invite them back with a curated set (Japanese set, or steak with wine). Lead with the menu, "
                                "not a discount: this segment rarely uses discounts.",
        ("Premium Foodies", 1): "Recommend the pairing that completes their order (green tea with sushi, wine with steak). "
                                "No discount needed.",
        ("Budget Students", 0): "A small combo-meal offer (wrap or burger, fries and a cola) is usually enough. "
                                "This segment responds to price.",
        ("Budget Students", 1): "No offer needed. Students almost always come back, so a discount here mostly gives margin away.",
    }
    generic = ("Send a win-back offer built on their segment's favourite items." if not reorder
               else "Likely to come back on their own. Focus on growing the basket rather than discounting.")
    return actions.get((name, reorder), generic)


def tab_drivers(profile, prob):
    if not HAS_DATA:
        st.write("Load diners.csv to compare this diner with a typical one.")
        return
    med = diners[CLF_FEATURES].median()
    rows = []
    for f in CLF_FEATURES:
        alt = dict(profile)
        alt[f] = med[f]
        rows.append(alt)
    alt_probs, _ = batch_predict(rows)
    eff = pd.DataFrame({"feature": [lab(f) for f in CLF_FEATURES], "effect": (prob - alt_probs) * 100,
                        "yours": [profile[f] for f in CLF_FEATURES], "typical": med.values})
    eff = eff.iloc[eff["effect"].abs().argsort()]
    fig = px.bar(eff, x="effect", y="feature", orientation="h", custom_data=["yours", "typical"],
                 color=np.where(eff["effect"] >= 0, "Raises the chance", "Lowers the chance"),
                 color_discrete_map={"Raises the chance": BASIL, "Lowers the chance": CHILI})
    fig.update_traces(hovertemplate="%{y}<br>This diner: %{customdata[0]}<br>Typical diner: %{customdata[1]}"
                                    "<br>Effect: %{x:+.0f} points<extra></extra>")
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title="Change in reorder chance vs. a typical value (percentage points)")
    fig.update_yaxes(title=None)
    show_chart(style(fig, h=340))
    learn("Each bar asks one question: if this diner had a typical value for this input (the median diner) "
          "and everything else stayed the same, how would their reorder chance change? Long bars are what matters most for this diner.")


def tab_why_segment(res):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Distance to each segment centre")
        dist = pd.DataFrame({"segment": [seg_names.get(i, f"Segment {i}") for i in range(len(res["dists"]))],
                             "distance": res["dists"]}).sort_values("distance")
        fig = px.bar(dist, x="distance", y="segment", orientation="h", color="segment",
                     color_discrete_map={s: seg_color(s) for s in dist["segment"]})
        fig.update_xaxes(title="Distance (scaled units, shorter = more alike)")
        fig.update_yaxes(title=None, categoryorder="array", categoryarray=dist["segment"].tolist()[::-1])
        show_chart(style(fig, h=260, legend=False))
    with c2:
        st.markdown(f"##### This diner vs. the average {res['name']} diner")
        if HAS_DATA:
            zs = segment_z()
            comp = pd.DataFrame({"feature": [lab(f) for f in FEATURES], "This diner": res["z"],
                                 "Segment average": zs.loc[res["name"]].values if res["name"] in zs.index else 0})
            fig = go.Figure()
            fig.add_bar(y=comp["feature"], x=comp["Segment average"], orientation="h", name="Segment average",
                        marker_color=seg_color(res["name"]), opacity=0.45)
            fig.add_scatter(y=comp["feature"], x=comp["This diner"], mode="markers", name="This diner",
                            marker=dict(color=INK, size=12, symbol="diamond"))
            fig.add_vline(x=0, line_color=MUTED, line_width=1)
            fig.update_xaxes(title="Standard deviations from the average diner")
            show_chart(style(fig, h=260))
    learn("K-Means puts a diner in the segment whose centre is closest, after scaling every input so that "
          "AED and order counts weigh the same. The shortest bar on the left wins.")


def tab_where(profile, res):
    if not HAS_DATA:
        st.write("Load diners.csv to see where this diner sits among everyone else.")
        return
    view = st.radio("View", ["2-D map", "Pick two features"], key="pv_view", horizontal=True)
    if view == "2-D map":
        pca = PCA(2, random_state=SEED).fit(X_SCALED)
        pts = pca.transform(X_SCALED)
        df = diners.assign(x=pts[:, 0], y=pts[:, 1])
        me = pca.transform(res["z"].reshape(1, -1))[0]
        xl, yl = "Map axis 1", "Map axis 2"
    else:
        c1, c2 = st.columns(2)
        fx = c1.selectbox("Horizontal axis", CLF_FEATURES, key="pv_x", format_func=lab)
        fy = c2.selectbox("Vertical axis", CLF_FEATURES, key="pv_y", format_func=lab)
        df = diners.assign(x=diners[fx], y=diners[fy])
        me = (profile[fx], profile[fy])
        xl, yl = lab(fx), lab(fy)
    fig = px.scatter(df, x="x", y="y", color="segment", hover_name="diner_id", opacity=0.55,
                     color_discrete_map={s: seg_color(s) for s in SEG_ORDER}, category_orders={"segment": SEG_ORDER},
                     hover_data={"x": False, "y": False, "reordered": True, "orders_per_month": True, "avg_order_value": True})
    fig.add_scatter(x=[me[0]], y=[me[1]], mode="markers+text", text=["This diner"], textposition="top center",
                    marker=dict(symbol="star", size=22, color=INK, line=dict(color="white", width=1.5)), name="This diner")
    fig.update_xaxes(title=xl)
    fig.update_yaxes(title=yl)
    show_chart(style(fig, h=440))
    learn("The 2-D map squeezes the five behaviour inputs into two axes (PCA) so you can see all diners at once. "
          "Click a segment in the legend to hide or show it.")


def tab_whatif(profile, res, thr):
    f = st.selectbox("Change one input and hold the rest", CLF_FEATURES, key="wf_feat", format_func=lab)
    lo, hi = SLIDER_RANGES[f]
    grid = np.linspace(lo, hi, 61) if f == "discount_rate" else np.unique(np.linspace(lo, hi, 61).round())
    rows = [{**profile, f: v} for v in grid]
    probs, segs = batch_predict(rows)
    df = pd.DataFrame({"value": grid, "prob": probs * 100, "segment": segs})
    fig = go.Figure()
    fig.add_scatter(x=df["value"], y=df["prob"], mode="lines", line=dict(color="#AEB9B4", width=2), showlegend=False,
                    hoverinfo="skip")
    for s in SEG_ORDER:
        d = df[df["segment"] == s]
        if len(d):
            fig.add_scatter(x=d["value"], y=d["prob"], mode="markers", name=s, marker=dict(color=seg_color(s), size=7),
                            hovertemplate=f"{lab(f)}: %{{x}}<br>Reorder chance: %{{y:.0f}}%<br>Segment: {s}<extra></extra>")
    fig.add_hline(y=thr * 100, line_dash="dash", line_color=INK, annotation_text=f"decision line {thr:.0%}",
                  annotation_position="top left")
    fig.add_vline(x=profile[f], line_color=INK, line_width=1, annotation_text="now", annotation_position="top right")
    fig.update_xaxes(title=lab(f))
    fig.update_yaxes(title="Reorder chance (%)", range=[0, 102])
    show_chart(style(fig, h=380))
    above = df["prob"] >= thr * 100
    flips = df.loc[above.ne(above.shift()) & (df.index > 0), "value"].tolist()
    seg_flips = df.loc[df["segment"].ne(df["segment"].shift()) & (df.index > 0), ["value", "segment"]]
    notes = []
    if flips:
        notes.append(f"The verdict flips at {lab(f).lower()} ≈ {', '.join(f'{v:g}' for v in flips)}.")
    else:
        notes.append(f"The verdict stays **{'Will reorder' if above.iloc[0] else 'At risk'}** across the whole range.")
    for _, r in seg_flips.iterrows():
        notes.append(f"From {r['value']:g} the diner is classed as **{r['segment']}**.")
    st.markdown(" ".join(notes))
    learn("Dot colours show which segment the diner would fall into at each value. A segment can change while "
          "the reorder chance barely moves, and the other way round: they come from two different models.")


def tab_bundles(name):
    if not HAS_DATA:
        st.write("Load orders.csv to see this segment's bundles.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"##### Strongest bundles for {name}")
        br = best_rules(mine_rules(name, 0.05)).head(6)
        if len(br):
            show_df(br[["if_bought", "then_also", "support", "confidence", "lift"]], hide_index=True,
                    column_config=rule_column_config())
        else:
            st.write("No rules pass the default filters for this segment.")
    with c2:
        st.markdown("##### Share of the segment's baskets")
        sh = item_shares(name).head(8).mul(100).rename("share").rename_axis("item").reset_index()
        sh["item"] = sh["item"].map(nice)
        fig = px.bar(sh, x="share", y="item", orientation="h", color_discrete_sequence=[seg_color(name)])
        fig.update_yaxes(title=None, autorange="reversed")
        fig.update_xaxes(title="% of baskets", ticksuffix="%")
        show_chart(style(fig, h=280, legend=False))


def rule_column_config():
    return {"if_bought": st.column_config.TextColumn("If the basket has"),
            "then_also": st.column_config.TextColumn("It often also has"),
            "support": st.column_config.NumberColumn("Support", format="%.3f", help="Share of all baskets with the whole combo"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f",
                                                          help="Of baskets with the first part, the share that also have the second"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2f", help="How many times more likely than chance"),
            "baskets": st.column_config.NumberColumn("Baskets", help="Number of baskets with the whole combo")}


# ------------------------------------------------------------ segments
@st.cache_data
def k_scan():
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_SCALED)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_SCALED, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_resource
def fit_k(k):
    return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_SCALED)


@st.cache_resource
def pca_fit(n):
    return PCA(n, random_state=SEED).fit(X_SCALED)


def page_segments():
    st.title("Segments")
    st.caption("Who are our diners? K-Means groups them on five behaviours: frequency, order value, order size, recency and discount use.")

    explore = st.toggle("Try a different number of segments", key="seg_explore",
                        help="Off: the saved model with named segments. On: refit K-Means with any k from 2 to 8.")
    if explore:
        k = st.slider("Number of segments (k)", 2, 8, key="seg_k")
        km = fit_k(k)
        labels = pd.Series([f"Cluster {i + 1}" for i in km.labels_], index=diners.index)
        order = [f"Cluster {i + 1}" for i in range(k)]
        centers = km.cluster_centers_
        colors = {s: EXTRA_COLORS[i % len(EXTRA_COLORS)] for i, s in enumerate(order)}
    else:
        k = kmeans.n_clusters
        labels = diners["segment"]
        order = SEG_ORDER
        centers = kmeans.cluster_centers_
        colors = {s: seg_color(s) for s in order}
    d = diners.assign(group=labels.values)

    scan = k_scan()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Elbow method")
        fig = px.line(scan, x="k", y="inertia", markers=True, color_discrete_sequence=[BASIL])
        fig.add_vline(x=k, line_dash="dash", line_color=CHILI)
        fig.update_xaxes(title="Number of segments (k)", dtick=1)
        fig.update_yaxes(title="Within-segment spread (inertia)")
        show_chart(style(fig, h=280, legend=False))
    with c2:
        st.markdown("##### Silhouette score")
        fig = px.line(scan, x="k", y="silhouette", markers=True, color_discrete_sequence=["#5A4A86"])
        fig.add_vline(x=k, line_dash="dash", line_color=CHILI)
        fig.update_xaxes(title="Number of segments (k)", dtick=1)
        fig.update_yaxes(title="Silhouette (higher = cleaner)")
        show_chart(style(fig, h=280, legend=False))
    sil = scan.set_index("k").loc[k, "silhouette"] if k in scan["k"].values else np.nan
    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    st.markdown(f"At **k = {k}** the silhouette score is **{sil:.3f}**. The best score is at **k = {best_k}**"
                + (", which is what the saved model uses." if best_k == kmeans.n_clusters else "."))
    learn("Inertia always falls as you add segments, so look for the bend where extra segments stop helping much (the elbow). "
          "Silhouette measures how clearly each diner belongs to its own segment rather than the next one: "
          "above 0.5 is a clean split. Turn on the toggle and slide k to watch the segments break apart.")

    st.divider()
    zdf = pd.DataFrame(X_SCALED, columns=FEATURES).assign(group=labels.values).groupby("group").mean().reindex(order)
    per_row = 4 if k > 4 else k
    for start in range(0, k, per_row):
        for col, g in zip(st.columns(per_row), order[start:start + per_row]):
            gd = d[d["group"] == g]
            with col:
                st.markdown(f'<div class="seg-name" style="color:{colors[g]}">{esc(g)}</div>', unsafe_allow_html=True)
                st.caption(f"{len(gd)} diners ({len(gd) / len(d):.0%})")
                st.write(f"{describe_z(zdf.loc[g])}.")
                st.markdown(f"Reorder rate **{gd['reordered'].mean():.0%}**")
                if not explore:
                    st.markdown(f"Top items: {', '.join(top_items(g, 4))}")

    st.divider()
    st.markdown("##### Map of diners")
    v1, v2 = st.columns([3, 1])
    view = v1.radio("View", ["2-D map", "Pick two features", "3-D view"], key="seg_view", horizontal=True)
    show_c = v2.toggle("Show segment centres", key="seg_centroids")
    cmap = dict(color_discrete_map=colors, category_orders={"group": order})
    hover = {"segment": False, "group": True, "reordered": True}
    if view == "2-D map":
        pca = pca_fit(2)
        pts = pca.transform(X_SCALED)
        fig = px.scatter(d.assign(x=pts[:, 0], y=pts[:, 1]), x="x", y="y", color="group", hover_name="diner_id",
                         opacity=0.6, hover_data={**hover, "x": False, "y": False}, **cmap)
        if show_c:
            cc = pca.transform(centers)
            fig.add_scatter(x=cc[:, 0], y=cc[:, 1], mode="markers", name="Centres",
                            marker=dict(symbol="x", size=16, color=INK, line=dict(width=2)))
        fig.update_xaxes(title=f"Map axis 1 ({pca.explained_variance_ratio_[0]:.0%} of variation)")
        fig.update_yaxes(title=f"Map axis 2 ({pca.explained_variance_ratio_[1]:.0%} of variation)")
        show_chart(style(fig, h=460))
    elif view == "Pick two features":
        a, b = st.columns(2)
        fx = a.selectbox("Horizontal axis", FEATURES + ["age", "total_spend"], key="seg_x", format_func=lab)
        fy = b.selectbox("Vertical axis", FEATURES + ["age", "total_spend"], key="seg_y", format_func=lab)
        fig = px.scatter(d, x=fx, y=fy, color="group", hover_name="diner_id", opacity=0.6, hover_data=hover,
                         labels={fx: lab(fx), fy: lab(fy)}, **cmap)
        if show_c and fx in FEATURES and fy in FEATURES:
            cc = pd.DataFrame(scaler.inverse_transform(centers), columns=FEATURES)
            fig.add_scatter(x=cc[fx], y=cc[fy], mode="markers", name="Centres",
                            marker=dict(symbol="x", size=16, color=INK, line=dict(width=2)))
        show_chart(style(fig, h=460))
    else:
        a, b, c = st.columns(3)
        opts = FEATURES + ["age", "total_spend"]
        fx = a.selectbox("X", opts, key="seg_x", format_func=lab)
        fy = b.selectbox("Y", opts, key="seg_y", format_func=lab)
        fz = c.selectbox("Z", opts, key="seg_z", format_func=lab)
        fig = px.scatter_3d(d, x=fx, y=fy, z=fz, color="group", hover_name="diner_id", opacity=0.7,
                            labels={fx: lab(fx), fy: lab(fy), fz: lab(fz)}, **cmap)
        fig.update_traces(marker=dict(size=3.5))
        show_chart(style(fig, h=560))
    learn("Click a name in the legend to hide that segment, double-click to show it alone. "
          "Drag to zoom; double-click the chart to reset. In 3-D, drag to rotate.")

    st.divider()
    st.markdown("##### Segment fingerprints")
    p1, p2 = st.columns(2)
    units = p1.radio("Values", ["Real units", "Standardised (z-scores)"], key="seg_units", horizontal=True)
    chart = p2.radio("Chart", ["Heatmap", "Radar", "Bars"], key="seg_chart", horizontal=True)
    real = d.groupby("group")[FEATURES].mean().reindex(order)
    if chart == "Heatmap":
        text = real.round(2).astype(str).values if units == "Real units" else zdf.round(2).astype(str).values
        fig = go.Figure(go.Heatmap(z=zdf.values, x=[lab(f) for f in FEATURES], y=order, text=text, texttemplate="%{text}",
                                   colorscale=[[0, "#3E7CB1"], [0.5, "#F7F7F7"], [1, CHILI]], zmid=0,
                                   colorbar=dict(title="z-score")))
        fig.update_yaxes(autorange="reversed")
        show_chart(style(fig, h=110 + 55 * k, legend=False))
        st.caption("Colour always shows how far a segment sits above (red) or below (blue) the average diner; "
                   "the numbers switch with the Values toggle.")
    elif chart == "Radar":
        fig = go.Figure()
        cats = [lab(f) for f in FEATURES]
        for g in order:
            vals = zdf.loc[g].tolist()
            fig.add_scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=g, fill="toself", opacity=0.55,
                                 line=dict(color=colors[g]))
        fig.update_layout(polar=dict(radialaxis=dict(gridcolor=GRID), angularaxis=dict(gridcolor=GRID), bgcolor="rgba(0,0,0,0)"))
        show_chart(style(fig, h=460))
        st.caption("The radar always uses z-scores so that every axis is on the same scale.")
    else:
        if units == "Real units":
            long = real.reset_index().melt(id_vars="group", var_name="feature", value_name="value")
            long["feature"] = long["feature"].map(lab)
            fig = px.bar(long, x="group", y="value", color="group", facet_col="feature", facet_col_wrap=5, **cmap)
            fig.update_yaxes(matches=None, showticklabels=True, title=None)
            fig.update_xaxes(showticklabels=False, title=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            show_chart(style(fig, h=330))
        else:
            long = zdf.reset_index().melt(id_vars="group", var_name="feature", value_name="z")
            long["feature"] = long["feature"].map(lab)
            fig = px.bar(long, x="feature", y="z", color="group", barmode="group", **cmap)
            fig.update_yaxes(title="Standard deviations from average")
            fig.update_xaxes(title=None)
            show_chart(style(fig, h=380))

    st.markdown("##### Spread within each segment")
    q1, q2 = st.columns([3, 1])
    f = q1.selectbox("Input", FEATURES + ["age", "total_spend"], key="seg_dist", format_func=lab)
    pts_on = q2.toggle("Show every diner", key="seg_points")
    fig = px.box(d, x="group", y=f, color="group", points="all" if pts_on else "outliers", labels={f: lab(f), "group": ""}, **cmap)
    show_chart(style(fig, h=360, legend=False))

    with st.expander("Diner table"):
        st.session_state.seg_filter = [s for s in st.session_state.seg_filter if s in order]
        pick = st.multiselect("Show segments", order, key="seg_filter", placeholder="All segments")
        tbl = d[d["group"].isin(pick)] if pick else d
        tbl = tbl.drop(columns=["cluster", "segment"]).rename(columns={"group": "segment"})
        show_df(tbl, hide_index=True)
        st.download_button("Download this table (CSV)", tbl.to_csv(index=False).encode(), "quickbite_segments.csv", "text/csv")


# ------------------------------------------------------------ basket rules
def page_baskets():
    st.title("Basket rules")
    st.caption("What do diners order together? Apriori treats every order as one basket and finds items that show up together more than chance.")

    scopes = ["All baskets"] + SEG_ORDER
    c = st.columns(4)
    scope = c[0].selectbox("Baskets from", scopes, key="b_scope", on_change=reset_support)
    min_sup = c[1].slider("Minimum support", 0.01, 0.30, step=0.01, key="b_sup",
                          help="Ignore combos that appear in fewer than this share of baskets.")
    min_conf = c[2].slider("Minimum confidence", 0.10, 1.00, step=0.05, key="b_conf")
    min_lift = c[3].slider("Minimum lift", 1.0, 10.0, step=0.1, key="b_lift")

    X = basket_matrix(scope)
    rules = mine_rules(scope, round(min_sup, 2))
    o = st.columns([1, 1, 1, 2])
    dedup = o[0].toggle("Hide mirrored rules", key="b_dedup", help="A → B and B → A share the same lift. Keep only the more confident one.")
    sort = o[1].selectbox("Rank by", ["lift", "confidence", "support"], key="b_sort")
    topn = o[2].slider("Rules to show", 5, 50, step=5, key="b_topn")
    st.session_state.b_items = [i for i in st.session_state.b_items if i in X.columns]
    items = o[3].multiselect("Only rules that mention", sorted(X.columns), key="b_items", placeholder="Any item",
                             format_func=nice)

    shown = best_rules(rules, min_conf, min_lift, dedup, sort) if len(rules) else rules
    if items and len(shown):
        mask = shown.apply(lambda r: any(i in r["antecedents"] + r["consequents"] for i in items), axis=1)
        shown = shown[mask]
    shown = shown.head(topn).reset_index(drop=True)

    m = st.columns(4)
    m[0].metric("Baskets", f"{len(X):,}")
    m[1].metric("Rules shown", f"{len(shown)}")
    m[2].metric("Strongest lift", f"{shown['lift'].max():.1f}" if len(shown) else "–")
    m[3].metric("Most common item", f"{nice(X.mean().idxmax())} ({X.mean().max():.0%})")
    learn("Support = how common the combo is. Confidence = of baskets with the first part, how many also have the second. "
          "Lift = how much more likely the second part becomes once the first is in the basket (1 means no link). "
          "A good bundle has high lift and enough support to matter.")

    if not len(shown):
        st.warning("No rules pass these filters. Lower the minimum support, confidence or lift, or clear the item filter.")
    else:
        t1, t2 = st.columns([3, 2])
        with t1:
            st.markdown("##### Ranked rules")
            show_df(shown[["if_bought", "then_also", "support", "confidence", "lift", "baskets"]], hide_index=True,
                    column_config=rule_column_config(), height=min(38 * len(shown) + 40, 420))
        with t2:
            st.markdown("##### Support vs. confidence")
            pl = shown.assign(rule=shown["if_bought"] + " → " + shown["then_also"])
            fig = px.scatter(pl, x="support", y="confidence", size="lift", color="lift", hover_name="rule",
                             color_continuous_scale=[[0, "#A9CBB7"], [1, "#1F4D38"]], size_max=26,
                             hover_data={"support": ":.3f", "confidence": ":.2f", "lift": ":.2f"})
            fig.update_traces(marker=dict(line=dict(color="white", width=1), opacity=0.9))
            fig.update_layout(coloraxis_colorbar=dict(title="lift"))
            show_chart(style(fig, h=380, legend=False))

        st.markdown("##### Explain a rule")
        labels = (shown["if_bought"] + " → " + shown["then_also"]).tolist()
        if st.session_state.b_explain not in labels:
            st.session_state.b_explain = labels[0]
        pick = st.selectbox("Pick a rule", labels, key="b_explain")
        r = shown.iloc[labels.index(pick)]
        A, B = list(r["antecedents"]), list(r["consequents"])
        hasA, hasB = X[A].all(axis=1), X[B].all(axis=1)
        n, nA, nB, nAB = len(X), int(hasA.sum()), int(hasB.sum()), int((hasA & hasB).sum())
        a_txt, b_txt = " + ".join(map(nice, A)), " + ".join(map(nice, B))
        e1, e2 = st.columns([3, 2])
        with e1:
            st.markdown(
                f"Out of **{n:,}** baskets, **{nA:,}** contain {a_txt}. Of those, **{nAB:,}** also contain {b_txt}. "
                f"That's the **confidence: {nAB / nA:.0%}**.\n\n"
                f"Across all baskets, {b_txt} shows up in only **{nB / n:.0%}**. So once {a_txt} is in the basket, "
                f"{b_txt} becomes **{r['lift']:.1f}× more likely** than usual. That's the **lift**.\n\n"
                f"The full combo appears in **{nAB / n:.1%}** of all baskets. That's the **support**.\n\n"
                f"**What to do:** when {a_txt} is in the cart, suggest {b_txt}, or sell them together as one bundle.")
        with e2:
            comp = pd.DataFrame({"group": ["All baskets", f"Baskets with {a_txt}"], "share": [nB / n * 100, nAB / nA * 100]})
            fig = px.bar(comp, x="group", y="share", text=comp["share"].map(lambda v: f"{v:.0f}%"),
                         color="group", color_discrete_sequence=["#AEB9B4", BASIL])
            fig.update_yaxes(title=f"% that contain {b_txt}", range=[0, 105])
            fig.update_xaxes(title=None)
            show_chart(style(fig, h=300, legend=False))

    st.divider()
    st.markdown("##### Which items pair up")
    pair = st.radio("Measure", ["Lift", "Bought together (count)", "Confidence (row → column)"], key="b_pair", horizontal=True)
    order = X.mean().sort_values(ascending=False).index.tolist()
    Xi = X[order].astype(int)
    C = Xi.T.dot(Xi).astype(float)
    diag = np.diag(C).copy()
    if pair == "Lift":
        mat = len(X) * C / np.outer(diag, diag)
    elif pair == "Bought together (count)":
        mat = C.copy()
    else:
        mat = C / diag[:, None]
    mat = np.where(np.eye(len(order), dtype=bool) | (C.values == 0), np.nan, mat)
    fmt = "{:.1f}" if pair == "Lift" else ("{:.0f}" if "count" in pair else "{:.2f}")
    text = [["" if np.isnan(v) else fmt.format(v) for v in row] for row in mat]
    names = [nice(i) for i in order]
    fig = go.Figure(go.Heatmap(z=mat, x=names, y=names, text=text, texttemplate="%{text}", textfont=dict(size=11),
                               colorscale=[[0, "#EEF4F0"], [1, "#1F4D38"]], hoverongaps=False,
                               colorbar=dict(title=pair.split(" (")[0]),
                               hovertemplate="%{y} with %{x}<br>%{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show_chart(style(fig, h=120 + 28 * len(order), legend=False))
    st.caption("Blank cells: never bought together in this scope.")
    learn("Dark cells are strong pairs. Switch the scope at the top to a single segment and the lifts drop: "
          "some of the overall pattern is simply each segment eating from its own menu.")

    st.markdown("##### How each segment's baskets differ")
    rows = []
    for s in SEG_ORDER:
        sh = item_shares(s)
        rows += [{"segment": s, "item": nice(i), "share": v * 100} for i, v in sh.items()]
    comp = pd.DataFrame(rows)
    item_order = [nice(i) for i in item_shares("All baskets").index]
    fig = px.bar(comp, x="item", y="share", color="segment", barmode="group",
                 color_discrete_map={s: seg_color(s) for s in SEG_ORDER},
                 category_orders={"item": item_order, "segment": SEG_ORDER})
    fig.update_yaxes(title="% of that segment's baskets", ticksuffix="%")
    fig.update_xaxes(title=None)
    show_chart(style(fig, h=380))


# ------------------------------------------------------------ reorder model
def page_model():
    st.title("Reorder model")
    st.caption("Who will come back? Try different models, move the decision line, and see what each kind of mistake costs.")

    c = st.columns([2, 1, 1, 1])
    name = c[0].selectbox("Model", MODEL_NAMES, key="m_model")
    test_size = c[1].slider("Test set size", 0.15, 0.40, step=0.05, key="m_test")
    cost_fa = c[2].number_input("Voucher cost (AED)", 0, 1000, step=5, key="m_fa",
                                help="A loyal diner flagged as at risk gets an offer they didn't need.")
    cost_miss = c[3].number_input("Lost diner cost (AED)", 0, 5000, step=25, key="m_miss",
                                  help="A churner the model missed: no win-back offer, and they're gone.")
    thr = st.slider("Decision threshold: predict 'will reorder' at or above", 0.05, 0.95, step=0.01, key="threshold")

    model, Xte, yte, proba = train(name, round(test_size, 2))
    cm = confusion_at(yte, proba, thr)
    n, churners = len(yte), int((yte == 0).sum())
    acc = (cm["caught"] + cm["kept"]) / n
    base_acc = max(yte.mean(), 1 - yte.mean())
    cost = cm["missed"] * cost_miss + cm["false_alarm"] * cost_fa
    base_cost = churners * cost_miss

    m = st.columns(5)
    m[0].metric("Accuracy", f"{acc:.1%}", f"{(acc - base_acc) * 100:+.1f} pts vs. baseline")
    m[1].metric("Churners caught", f"{cm['caught']} of {churners}", help=f"{cm['caught'] / churners:.0%} of the churners in the test set")
    m[2].metric("Missed churners", cm["missed"], help="The expensive mistake")
    m[3].metric("False alarms", cm["false_alarm"], help="The cheap mistake")
    m[4].metric("Cost of mistakes", f"AED {cost:,.0f}", f"{cost - base_cost:+,.0f} vs. no model", delta_color="inverse",
                help=f"With no model you'd lose all {churners} churners: AED {base_cost:,.0f}.")
    st.caption(f"Test set: {n} diners the model never saw ({churners} churned). Baseline = always guess 'will reorder' "
               f"({base_acc:.0%} accurate, catches no churners).")
    learn("Missing a churner costs a customer; a false alarm costs one voucher. Raising the threshold flags more diners as "
          "at risk: fewer missed churners, more wasted vouchers. Set your own costs and let the cost chart find the cheapest line.")

    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("##### Confusion matrix")
        z = [[cm["caught"], cm["missed"]], [cm["false_alarm"], cm["kept"]]]
        text = [[f"Caught churner<br><b>{cm['caught']}</b>", f"Missed churner<br><b>{cm['missed']}</b>"],
                [f"False alarm<br><b>{cm['false_alarm']}</b>", f"Correctly kept<br><b>{cm['kept']}</b>"]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted at risk", "Predicted will reorder"], y=["Actually churned", "Actually reordered"],
                                   text=text, texttemplate="%{text}", textfont=dict(size=14), showscale=False,
                                   colorscale=[[0, "#F2F6F3"], [1, "#8DB8A0"]], hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        show_chart(style(fig, h=320, legend=False))
    with b:
        view = st.radio("Chart", ["Probability split", "Cost vs. threshold", "ROC curve"], key="m_view", horizontal=True)
        if view == "Probability split":
            df = pd.DataFrame({"prob": proba * 100, "actual": np.where(yte == 1, "Actually reordered", "Actually churned")})
            fig = px.histogram(df, x="prob", color="actual", barmode="stack", opacity=0.9,
                               color_discrete_map={"Actually reordered": BASIL, "Actually churned": CHILI})
            fig.update_traces(xbins=dict(start=0, end=100, size=5))
            fig.add_vline(x=thr * 100, line_dash="dash", line_color=INK, annotation_text="decision line")
            fig.update_xaxes(title="Predicted reorder chance (%)", range=[0, 100])
            fig.update_yaxes(title="Diners")
            show_chart(style(fig, h=330))
            st.caption("Red bars right of the line are missed churners; green bars left of it are false alarms.")
        elif view == "Cost vs. threshold":
            ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
            costs = [confusion_at(yte, proba, t)["missed"] * cost_miss + confusion_at(yte, proba, t)["false_alarm"] * cost_fa for t in ts]
            best_t = float(ts[int(np.argmin(costs))])
            fig = px.line(x=ts, y=costs, color_discrete_sequence=[BASIL])
            fig.add_vline(x=thr, line_dash="dash", line_color=INK, annotation_text="current")
            fig.add_scatter(x=[best_t], y=[min(costs)], mode="markers", marker=dict(size=12, color=CHILI), name="cheapest")
            fig.update_xaxes(title="Decision threshold")
            fig.update_yaxes(title="Cost of mistakes (AED)")
            show_chart(style(fig, h=300, legend=False))
            st.button(f"Use the cheapest threshold ({best_t:.2f})", on_click=set_threshold, args=(best_t,))
        else:
            fpr, tpr, _ = roc_curve(yte, proba)
            auc = roc_auc_score(yte, proba)
            pt_tpr = cm["kept"] / max(cm["kept"] + cm["false_alarm"], 1)
            pt_fpr = cm["missed"] / max(cm["missed"] + cm["caught"], 1)
            fig = px.area(x=fpr, y=tpr, color_discrete_sequence=[BASIL])
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=MUTED), name="random guess")
            fig.add_scatter(x=[pt_fpr], y=[pt_tpr], mode="markers", marker=dict(size=12, color=CHILI), name="current threshold")
            fig.update_xaxes(title="Churners wrongly predicted to reorder (false positive rate)", range=[0, 1])
            fig.update_yaxes(title="Reorderers correctly predicted (true positive rate)", range=[0, 1.02])
            show_chart(style(fig, h=330, legend=False))
            st.caption(f"Area under the curve (ROC-AUC) = {auc:.3f}. 0.5 is a coin flip, 1.0 is perfect.")

    st.divider()
    d1, d2 = st.columns([3, 2], gap="large")
    with d1:
        st.markdown("##### How the model decides")
        est = model[-1] if hasattr(model, "steps") else model
        if hasattr(est, "coef_"):
            s = pd.Series(est.coef_[0], index=[lab(f) for f in CLF_FEATURES]).sort_values()
            fig = px.bar(x=s.values, y=s.index, orientation="h", color=np.where(s.values >= 0, "Raises odds", "Lowers odds"),
                         color_discrete_map={"Raises odds": BASIL, "Lowers odds": CHILI})
            fig.update_xaxes(title="Coefficient on scaled inputs")
        elif hasattr(est, "feature_importances_"):
            s = pd.Series(est.feature_importances_, index=[lab(f) for f in CLF_FEATURES]).sort_values()
            fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=[BASIL])
            fig.update_xaxes(title="Importance")
        else:
            fig = None
        if fig is not None:
            fig.update_yaxes(title=None)
            show_chart(style(fig, h=320, legend=hasattr(est, "coef_")))
    with d2:
        st.markdown("##### Where the mistakes happen")
        seg_te = diners.loc[Xte.index, "segment"].values
        pred = proba >= thr
        err = pd.DataFrame({"segment": seg_te, "missed": (yte == 0) & pred, "false_alarm": (yte == 1) & ~pred})
        err = err.groupby("segment")[["missed", "false_alarm"]].sum().reindex(SEG_ORDER).fillna(0).reset_index()
        long = err.melt(id_vars="segment", var_name="mistake", value_name="diners")
        long["mistake"] = long["mistake"].map({"missed": "Missed churners", "false_alarm": "False alarms"})
        fig = px.bar(long, x="segment", y="diners", color="mistake", barmode="group",
                     color_discrete_map={"Missed churners": CHILI, "False alarms": "#D9962B"})
        fig.update_xaxes(title=None)
        show_chart(style(fig, h=320))
    learn("Recency (days since last order) is the strongest warning sign. Most mistakes land in the segment where "
          "reordering is close to a coin flip; students and families are easy calls.")

    if st.toggle("Compare all models side by side", key="m_compare"):
        rows = []
        for nm in MODEL_NAMES:
            _, _, y_, p_ = train(nm, round(test_size, 2))
            c_ = confusion_at(y_, p_, thr)
            rows.append({"model": nm, "accuracy": (c_["caught"] + c_["kept"]) / len(y_), "ROC-AUC": roc_auc_score(y_, p_),
                         "churners caught": c_["caught"] / max(c_["caught"] + c_["missed"], 1),
                         "missed churners": c_["missed"], "false alarms": c_["false_alarm"],
                         "cost (AED)": c_["missed"] * cost_miss + c_["false_alarm"] * cost_fa})
        show_df(pd.DataFrame(rows), hide_index=True, column_config={
            "accuracy": st.column_config.NumberColumn(format="%.3f"), "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "churners caught": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            "cost (AED)": st.column_config.NumberColumn(format="%.0f")})
        st.caption(f"All models use the same test set and the current threshold ({thr:.2f}).")


# ------------------------------------------------------------ at-risk list
def page_atrisk():
    with st.sidebar:
        st.subheader("Decision rule")
        st.slider("Reorder threshold", 0.05, 0.95, step=0.01, key="threshold")
    thr = st.session_state.threshold
    st.title("At-risk list")
    st.caption("Every diner scored by the saved reorder model, with the items to build a win-back offer around.")

    probs, _ = batch_predict(diners[CLF_FEATURES].to_dict("records"))
    df = diners.assign(prob=probs * 100, verdict=np.where(probs >= thr, "Will reorder", "At risk"))
    df["win_back_items"] = df["segment"].map(lambda s: ", ".join(nice(i) for i in reco.get(s, [])[:3]))

    c = st.columns([2, 2, 1, 2])
    st.session_state.r_segs = [s for s in st.session_state.r_segs if s in SEG_ORDER]
    segs = c[0].multiselect("Segments", SEG_ORDER, key="r_segs", placeholder="All segments")
    lo, hi = c[1].slider("Reorder chance between", 0.0, 1.0, step=0.05, key="r_range")
    only = c[2].toggle("At-risk only", key="r_only")
    sort = c[3].selectbox("Sort by", ["Lowest reorder chance", "Highest monthly value", "Longest since last order"], key="r_sort")

    v = df[(df["prob"] >= lo * 100) & (df["prob"] <= hi * 100)]
    if segs:
        v = v[v["segment"].isin(segs)]
    if only:
        v = v[v["verdict"] == "At risk"]
    v = v.sort_values({"Lowest reorder chance": "prob", "Highest monthly value": "monthly_value",
                       "Longest since last order": "days_since_last_order"}[sort],
                      ascending=sort == "Lowest reorder chance")

    m = st.columns(4)
    risk = v[v["verdict"] == "At risk"]
    m[0].metric("Diners in view", f"{len(v):,}")
    m[1].metric("Flagged at risk", f"{len(risk):,}")
    m[2].metric("Monthly order value at risk", f"AED {risk['monthly_value'].sum():,.0f}")
    m[3].metric("Average reorder chance", f"{v['prob'].mean():.0f}%" if len(v) else "–")
    learn("Use this list weekly: score every diner, send the at-risk ones an offer built on their segment's favourite items. "
          "The model learned from most of these diners, so scores here look more confident than they would on new data.")

    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("##### Reorder chance vs. days since last order")
        fig = px.scatter(df if not segs else df[df["segment"].isin(segs)], x="days_since_last_order", y="prob",
                         color="segment", hover_name="diner_id", opacity=0.7, hover_data={"verdict": True},
                         color_discrete_map={s: seg_color(s) for s in SEG_ORDER}, category_orders={"segment": SEG_ORDER},
                         labels={"days_since_last_order": "Days since last order", "prob": "Reorder chance (%)"})
        fig.add_hline(y=thr * 100, line_dash="dash", line_color=INK, annotation_text="decision line")
        show_chart(style(fig, h=380))
    with b:
        st.markdown("##### At risk by segment")
        cnt = df.groupby(["segment", "verdict"]).size().reset_index(name="diners")
        fig = px.bar(cnt, x="segment", y="diners", color="verdict", category_orders={"segment": SEG_ORDER},
                     color_discrete_map={"Will reorder": BASIL, "At risk": CHILI})
        fig.update_xaxes(title=None)
        show_chart(style(fig, h=380))

    st.markdown("##### The list")
    out = v[["diner_id", "segment", "prob", "verdict", "days_since_last_order", "orders_per_month",
             "avg_order_value", "monthly_value", "win_back_items"]]
    show_df(out, hide_index=True, height=420, column_config={
        "diner_id": "Diner", "segment": "Segment", "verdict": "Verdict",
        "prob": st.column_config.ProgressColumn("Reorder chance", min_value=0, max_value=100, format="%.0f%%"),
        "days_since_last_order": st.column_config.NumberColumn("Days since last order"),
        "orders_per_month": st.column_config.NumberColumn("Orders / month"),
        "avg_order_value": st.column_config.NumberColumn("AED / order", format="%.0f"),
        "monthly_value": st.column_config.NumberColumn("Monthly value (AED)", format="%.0f"),
        "win_back_items": "Build the offer around"})
    st.download_button("Download this list (CSV)", out.to_csv(index=False).encode(), "quickbite_at_risk.csv", "text/csv")


# ------------------------------------------------------------ takeaways
def page_takeaways():
    st.title("Takeaways")
    st.caption("What QuickBite should do next, with the numbers recalculated live from the data and models.")

    g = diners.groupby("segment").agg(diners=("diner_id", "size"), reorder_rate=("reordered", "mean"),
                                      aov=("avg_order_value", "mean"), discount=("discount_rate", "mean"),
                                      value_per_diner=("monthly_value", "mean")).reindex(SEG_ORDER)
    g["value_lost"] = diners[diners["reordered"] == 0].groupby("segment")["monthly_value"].sum().reindex(SEG_ORDER).fillna(0)
    g["value_kept"] = diners[diners["reordered"] == 1].groupby("segment")["monthly_value"].sum().reindex(SEG_ORDER).fillna(0)
    focus = g["value_lost"].idxmax()
    swing = (g["reorder_rate"] - 0.5).abs().idxmin()
    loyal = g["reorder_rate"].idxmax()

    def bundle_text(seg):
        br = best_rules(mine_rules(seg, 0.05))
        if not len(br):
            return "no strong bundle"
        b = br.iloc[0]
        return f"{b['if_bought']} + {b['then_also']} ({b['confidence']:.0%} confidence)"

    _, _, y_, p_ = train(SAVED_MODEL, 0.25)
    c_ = confusion_at(y_, p_, st.session_state.threshold)
    catch = c_["caught"] / max(c_["caught"] + c_["missed"], 1)

    st.markdown(f'<div class="hero" style="font-size:clamp(1.7rem,3.2vw,2.5rem)">Win back {esc(focus)} first.</div>',
                unsafe_allow_html=True)
    body, _ = st.columns([3, 1])
    body.markdown(
        f"**{focus}** hold the biggest leak: only **{g.loc[focus, 'reorder_rate']:.0%}** reorder, which leaves "
        f"**AED {g.loc[focus, 'value_lost']:,.0f}** of monthly order value with diners who have gone. "
        f"Promote their own bundle as a meal deal: **{bundle_text(focus)}**.\n\n"
        f"**{swing}** are the swing segment ({g.loc[swing, 'reorder_rate']:.0%} reorder) and are worth "
        f"AED {g.loc[swing, 'value_per_diner']:,.0f} a month per diner. Keep them with curated sets such as "
        f"**{bundle_text(swing)}**, not discounts.\n\n"
        f"Run the reorder model weekly as an early-warning list. At the current threshold "
        f"({st.session_state.threshold:.2f}) it catches **{catch:.0%}** of churners on unseen diners, and its false alarms "
        f"only cost a voucher.\n\n"
        f"**{loyal}** already reorder {g.loc[loyal, 'reorder_rate']:.0%} of the time while using a discount on "
        f"{g.loc[loyal, 'discount']:.0%} of orders. Test pulling some of that discount spend back.")

    if st.toggle("Show the numbers behind it", key="t_numbers"):
        long = g[["value_kept", "value_lost"]].reset_index().melt(id_vars="segment", var_name="status", value_name="aed")
        long["status"] = long["status"].map({"value_kept": "Diners who reordered", "value_lost": "Diners who churned"})
        fig = px.bar(long, x="segment", y="aed", color="status", color_discrete_map={"Diners who reordered": BASIL,
                                                                                       "Diners who churned": CHILI})
        fig.update_yaxes(title="Monthly order value (AED)")
        fig.update_xaxes(title=None)
        show_chart(style(fig, h=360))
        show_df(g.reset_index().rename(columns={"aov": "avg order value", "discount": "discount use",
                                                "value_per_diner": "monthly value per diner",
                                                "value_lost": "monthly value lost", "value_kept": "monthly value kept"}),
                hide_index=True, column_config={
                    "reorder_rate": st.column_config.NumberColumn("reorder rate", format="%.2f"),
                    "avg order value": st.column_config.NumberColumn(format="%.0f"),
                    "discount use": st.column_config.NumberColumn(format="%.2f"),
                    "monthly value per diner": st.column_config.NumberColumn(format="%.0f"),
                    "monthly value lost": st.column_config.NumberColumn(format="%.0f"),
                    "monthly value kept": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Monthly order value = orders per month × average order value.")


# ------------------------------------------------------------ router
{"🏠 Overview": page_overview, "🔮 Diner predictor": page_predictor, "🧩 Segments": page_segments,
 "🛒 Basket rules": page_baskets, "🔁 Reorder model": page_model, "🚨 At-risk list": page_atrisk,
 "💡 Takeaways": page_takeaways}[st.session_state.page]()

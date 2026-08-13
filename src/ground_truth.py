"""Construct pre-specified event anchors for real financial data.

IMPORTANT TERMINOLOGY NOTE:

These event dates are NOT exact true change points in any mathematical
sense. Real financial regime shifts have no precise unambiguous onset
time. We use these dates as PRE-SPECIFIED EVENT ANCHORS for
event-window validation: a detection (offline or alarm) is considered
aligned with an anchor if it falls within a fixed tolerance window
around the anchor.

In paper writing, refer to these as 'event anchors' or 'pre-specified
event windows', NOT 'ground truth' or 'true changepoints'. The
'ground_truth' file/function names are kept for code-internal
clarity but the paper narrative must use the weaker, more accurate
framing.

Two categories included in this minimal version:
    1. NBER recession start dates. These are monthly peak dates from
       the NBER Business Cycle Dating Committee and are mapped to the
       first available trading day of the corresponding month.
    2. Single-day SPX log return below CRASH_THRESHOLD (purely
       data-driven, mechanically defined).

Both categories are mechanical and reproducible from public data.
The crash-event set is intentionally conservative: it will miss slow
regime transitions like the dot-com decline and the early Lehman
deterioration. This conservatism is by design to avoid editorial
selection of events.

Future extensions (deferred to Medium/Loose anchor sets):
    3. Fed emergency actions (FOMC non-scheduled meetings)
    4. Major geopolitical events
"""
import pandas as pd


# ====================================================================
# NBER recession start dates
# ====================================================================
# Source: https://www.nber.org/research/business-cycle-dating
# These are NBER-defined peak months, mapped to first trading day of
# the month at evaluation time. Stable, public, mechanically defined.

NBER_RECESSION_STARTS = [
    "1990-07-01",  # Gulf War recession
    "2001-03-01",  # dot-com recession
    "2007-12-01",  # Great Recession
    "2020-02-01",  # COVID recession
]


# ====================================================================
# Threshold for crash day (mechanical, no editorial judgment)
# ====================================================================
CRASH_THRESHOLD = -0.07   # -7% single-day log return


# ====================================================================
# Construction
# ====================================================================

def build_ground_truth(spx_df, crash_threshold=CRASH_THRESHOLD,
                       include_nber=True, include_crashes=True,
                       min_gap_days=None):
    """Build event anchor dict from SPX data.

    Args:
        spx_df: DataFrame with DatetimeIndex and 'log_return' column.
        crash_threshold: log return below which a day counts as crash.
        include_nber: include NBER recession starts.
        include_crashes: include single-day crash dates.
        min_gap_days: if set, merge anchors closer than this many days
            into single episodes (keeps the earliest in each cluster).
            Useful for evaluation when event windows would overlap.
            Default None = no merging.

    Returns:
        dict: {date_str: label} for all anchors in sample period.
    """
    events = {}

    if include_nber:
        for date_str in NBER_RECESSION_STARTS:
            ts = pd.Timestamp(date_str)
            if spx_df.index.min() <= ts <= spx_df.index.max():
                events[date_str] = "NBER recession start"

    if include_crashes:
        crash_mask = spx_df["log_return"] < crash_threshold
        crash_days = spx_df.index[crash_mask]
        for ts in crash_days:
            date_str = ts.strftime("%Y-%m-%d")
            # Defensive: in case of duplicate index entries
            log_ret = float(spx_df.loc[ts, "log_return"])
            label = f"Crash day ({log_ret:.3f})"
            if date_str in events:
                events[date_str] = events[date_str] + " + " + label
            else:
                events[date_str] = label

    if min_gap_days is not None and len(events) > 1:
        events = _apply_min_gap(events, min_gap_days)

    return events


def _apply_min_gap(events_dict, min_gap_days):
    """Merge anchors closer than min_gap_days into single episodes.

    Keeps the earliest anchor in each cluster. Labels of merged
    anchors are appended.
    """
    sorted_items = sorted(events_dict.items(),
                          key=lambda kv: pd.Timestamp(kv[0]))
    kept = {}
    last_ts = None
    last_key = None

    for date_str, label in sorted_items:
        ts = pd.Timestamp(date_str)
        if last_ts is None or (ts - last_ts).days >= min_gap_days:
            kept[date_str] = label
            last_ts = ts
            last_key = date_str
        else:
            # Merge into the earlier anchor's label
            kept[last_key] = kept[last_key] + " + " + label
    return kept


def ground_truth_dates(events_dict):
    """Convert events dict to sorted list of Timestamps."""
    return sorted(pd.Timestamp(d) for d in events_dict.keys())


def ground_truth_indices(events_dict, spx_df):
    """Convert events dict to sorted unique list of row indices.

    For anchor dates on non-trading days (e.g., NBER monthly peaks),
    maps to the first available trading day at or after the anchor.
    Returns sorted unique indices; if two distinct anchors map to the
    same trading day (rare), they are deduplicated.
    """
    indices = []
    for date_str in events_dict.keys():
        ts = pd.Timestamp(date_str)
        future_mask = spx_df.index >= ts
        if not future_mask.any():
            continue  # date after sample end
        first_idx = spx_df.index[future_mask][0]
        row_idx = spx_df.index.get_loc(first_idx)
        indices.append(row_idx)
    return sorted(set(indices))


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loaders import download_spx

    df = download_spx()

    print("=" * 60)
    print("Event anchors (Strict set: NBER + crashes < -7%)")
    print("=" * 60)
    events = build_ground_truth(df)
    print(f"Sample period: {df.index.min().date()} to "
          f"{df.index.max().date()}")
    print(f"Total anchors: {len(events)}")
    print()
    for date_str in sorted(events.keys()):
        print(f"  {date_str}  {events[date_str]}")
    print()

    indices = ground_truth_indices(events, df)
    print(f"Row indices: {indices}")
    print(f"Unique indices: {len(indices)}")
    print()

    # Demonstrate min_gap_days option
    print("=" * 60)
    print("Event anchors with min_gap_days=30 (episode-level)")
    print("=" * 60)
    events_grouped = build_ground_truth(df, min_gap_days=30)
    print(f"Total episodes: {len(events_grouped)}")
    for date_str in sorted(events_grouped.keys()):
        print(f"  {date_str}  {events_grouped[date_str]}")

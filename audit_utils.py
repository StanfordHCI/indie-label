"""
UTILS FILE
"""
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import mne

from surprise import Dataset, Reader, SVD, accuracy, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import math
import altair as alt
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer, util
import torch
from bertopic import BERTopic

########################################
# PRE-LOADING

YOUR_COLOR = '#6CADFD'
OTHER_USERS_COLOR = '#ccc'
BINS = [0, 0.5, 1.5, 2.5, 3.5, 4]
BIN_LABELS = ['0: Not at all toxic', '1: Slightly toxic', '2: Moderately toxic', '3: Very toxic', '4: Extremely toxic']
TOXIC_THRESHOLD = 2.0

alt.renderers.enable('altair_saver', fmts=['vega-lite', 'png'])

# Data-loading
module_dir = "./"
perf_dir = f"data/perf/"

# # TEMP reset
# with open(os.path.join(module_dir, "./data/all_model_names.pkl"), "wb") as f:
#     all_model_names = []
#     pickle.dump(all_model_names, f)
# with open(f"./data/users_to_models.pkl", "wb") as f:
#     users_to_models = {}
#     pickle.dump(users_to_models, f)


with open(os.path.join(module_dir, "data/ids_to_comments.pkl"), "rb") as f:
    ids_to_comments = pickle.load(f)
with open(os.path.join(module_dir, "data/comments_to_ids.pkl"), "rb") as f:
    comments_to_ids = pickle.load(f)

all_model_names = sorted([name for name in os.listdir(os.path.join(perf_dir)) if os.path.isdir(os.path.join(perf_dir, name))])
comments_grouped_full_topic_cat = pd.read_pickle("data/comments_grouped_full_topic_cat2_persp.pkl")
sys_eval_df = pd.read_pickle(os.path.join(module_dir, "data/split_data/sys_eval_df.pkl"))
train_df = pd.read_pickle(os.path.join(module_dir, "data/split_data/train_df.pkl"))
train_df_ids = train_df["item_id"].unique().tolist()
model_eval_df = pd.read_pickle(os.path.join(module_dir, "data/split_data/model_eval_df.pkl"))
ratings_df_full = pd.read_pickle(os.path.join(module_dir, "data/ratings_df_full.pkl"))

worker_info_df = pd.read_pickle("./data/worker_info_df.pkl")

with open(f"./data/users_to_models.pkl", "rb") as f:
    users_to_models = pickle.load(f)

with open("data/perf_1000_topics.pkl", "rb") as f:
    perf_1000_topics = pickle.load(f)
with open("data/perf_1000_tox_cat.pkl", "rb") as f:
    perf_1000_tox_cat = pickle.load(f)
with open("data/perf_1000_tox_severity.pkl", "rb") as f:
    perf_1000_tox_severity = pickle.load(f)
with open("data/user_perf_metrics.pkl", "rb") as f:
    user_perf_metrics = pickle.load(f)

topic_ids = comments_grouped_full_topic_cat.topic_id
topics = comments_grouped_full_topic_cat.topic
topic_ids_to_topics = {topic_ids[i]: topics[i] for i in range(len(topic_ids))}
topics_to_topic_ids = {topics[i]: topic_ids[i] for i in range(len(topic_ids))}
unique_topics_ids = sorted(comments_grouped_full_topic_cat.topic_id.unique())
unique_topics = [topic_ids_to_topics[topic_id] for topic_id in range(len(topic_ids_to_topics) - 1)]

def get_toxic_threshold():
    return TOXIC_THRESHOLD

def get_all_model_names(user=None):
    if (user is None) or (user not in users_to_models):
        all_model_names = sorted([name for name in os.listdir(os.path.join(perf_dir)) if os.path.isdir(os.path.join(perf_dir, name))])
        return all_model_names
    else:
        # Fetch the user's models
        user_models = users_to_models[user]
        user_models.sort()
        return user_models

def get_unique_topics():
    return unique_topics

def get_large_clusters(min_n):
    counts_df = comments_grouped_full_topic_cat.groupby(by=["topic_id"]).size().reset_index(name='counts')
    counts_df = counts_df[counts_df["counts"] >= min_n]
    return [topic_ids_to_topics[t_id] for t_id in sorted(counts_df["topic_id"].tolist()[1:])]

def get_ids_to_comments():
    return ids_to_comments

def get_workers_in_group(sel_gender, sel_race, sel_relig, sel_pol, sel_lgbtq):
    df = worker_info_df.copy()
    if sel_gender != "null":
        df = df[df["gender"] == sel_gender]
    if sel_relig != "null":
        df = df[df["religion_important"] == sel_relig]
    if sel_pol != "null":
        df = df[df["political_affilation"] == sel_pol]
    if sel_lgbtq != "null":
        if sel_lgbtq == "LGBTQ+":
            df = df[(df["lgbtq_status"] == "Homosexual") | (df["lgbtq_status"] == "Bisexual")]  
        else:
            df = df[df["lgbtq_status"] == "Heterosexual"]  
    if sel_race != "":
        df = df.dropna(subset=['race'])
        for r in sel_race:
            # Filter to rows with the indicated race
            df = df[df["race"].str.contains(r)]
    return df, len(df)

readable_to_internal = {
    "Mean Absolute Error (MAE)": "MAE",
    "Root Mean Squared Error (RMSE)": "RMSE",
    "Mean Squared Error (MSE)": "MSE",
    "Average rating difference": "avg_diff",
    "Topic": "topic",
    "Toxicity Category": "toxicity_category",
    "Toxicity Severity": "toxicity_severity",
}
internal_to_readable = {v: k for k, v in readable_to_internal.items()}

# Embeddings for neighbor retrieval
model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
with open("./data/comments.pkl", "rb") as f:
    comments = pickle.load(f)
embeddings = torch.load("./data/embeddings/21_10_embeddings.pt")

# Perspective API recalibration
def recalib_v1(s):
    # convert Perspective score to 0-4 toxicity score
    # map 0 persp to 0 (not at all toxic); 0.5 persp to 1 (slightly toxic), 1.0 persp to 4 (extremely toxic)
    if s < 0.5:
        return (s * 2.)
    else:
        return ((s - 0.5) * 6.) + 1

def recalib_v2(s):
    # convert Perspective score to 0-4 toxicity score
    # just 4x the perspective score 
    return (s * 4.)

comments_grouped_full_topic_cat["rating_avg_orig"] = comments_grouped_full_topic_cat["rating"]
comments_grouped_full_topic_cat["rating"] = [recalib_v2(score) for score in comments_grouped_full_topic_cat["persp_score"].tolist()]

def get_comments_grouped_full_topic_cat():
    return comments_grouped_full_topic_cat

########################################
# General utils
def get_metric_ind(metric):
    if metric == "MAE":
        ind = 0
    elif metric == "MSE":
        ind = 1
    elif metric == "RMSE":
        ind = 2
    elif metric == "avg_diff":
        ind = 3
    return ind

def my_bootstrap(vals, n_boot, alpha):
    bs_samples = []
    sample_size = len(vals)
    for i in range(n_boot):
        samp = resample(vals, n_samples=sample_size)
        bs_samples.append(np.median(samp))
        
    p = ((1.0 - alpha) / 2.0) * 100
    ci_low = np.percentile(bs_samples, p)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    ci_high = np.percentile(bs_samples, p)
    return bs_samples, (ci_low, ci_high)

########################################
# GET_AUDIT utils
def other_users_perf(perf_metrics, metric, user_metric, alpha=0.95, n_boot=501):
    ind = get_metric_ind(metric)
    
    metric_vals = [metric_vals[ind] for metric_vals in perf_metrics.values()]
    metric_avg = np.median(metric_vals)
    
    # Future: use provided sample to perform bootstrap sampling
    ci_1 = mne.stats.bootstrap_confidence_interval(np.array(metric_vals), ci=alpha, n_bootstraps=n_boot, stat_fun="median")
    
    bs_samples, ci = my_bootstrap(metric_vals, n_boot, alpha)
    
    # Get user's percentile
    percentile = stats.percentileofscore(bs_samples, user_metric)
    
    return metric_avg, ci, percentile, metric_vals

def plot_metric_histogram(metric, user_metric, other_metric_vals, n_bins=10):
    hist, bin_edges = np.histogram(other_metric_vals, bins=n_bins, density=False)
    data = pd.DataFrame({
        "bin_min": bin_edges[:-1],
        "bin_max": bin_edges[1:],
        "bin_count": hist,
        "user_metric": [user_metric for i in range(len(hist))]
    })
    base = alt.Chart(data)

    bar = base.mark_bar(color=OTHER_USERS_COLOR).encode(
        x=alt.X("bin_min", bin="binned", title=internal_to_readable[metric]),
        x2='bin_max',
        y=alt.Y("bin_count", title="Number of users"),
        tooltip=[
            alt.Tooltip('bin_min', title=f'{metric} bin min', format=".2f"),
            alt.Tooltip('bin_max', title=f'{metric} bin max', format=".2f"),
            alt.Tooltip('bin_count', title=f'Number of OTHER users', format=","),
        ]
    )

    rule = base.mark_rule(color=YOUR_COLOR).encode(
        x = "mean(user_metric):Q",
        size=alt.value(2),
        tooltip=[
            alt.Tooltip('mean(user_metric)', title=f'{metric} with YOUR labels', format=".2f"),
        ]
    )

    return (bar + rule).interactive()

def get_toxicity_severity_bins(perf_metric, user_df, other_dfs, bins=BINS, bin_labels=BIN_LABELS, ci=0.95, n_boot=501):
    # Note: not using other_dfs anymore
    y_user = []
    y_other = []
    used_bins = []
    other_ci_low = []
    other_ci_high = []
    for severity_i in range(len(bin_labels)):
        metric_others = [metrics[get_metric_ind(perf_metric)] for metrics in perf_1000_tox_severity[severity_i].values() if metrics[get_metric_ind(perf_metric)]]
        ci_low, ci_high = mne.stats.bootstrap_confidence_interval(np.array(metric_others), ci=ci, n_bootstraps=n_boot, stat_fun='median')
        metric_other = np.median(metric_others)
        
        cur_user_df = user_df[user_df["prediction_bin"] == severity_i]
        y_true_user = cur_user_df.pred.to_numpy()  # user's label
        y_pred = cur_user_df.rating_avg.to_numpy()  # system's label (avg)
        
        if len(y_true_user) > 0:
            used_bins.append(bin_labels[severity_i])
            metric_user = calc_metric_user(y_true_user, y_pred, perf_metric)
            y_user.append(metric_user)
            y_other.append(metric_other)
            other_ci_low.append(ci_low)
            other_ci_high.append(ci_high)
            
    return y_user, y_other, used_bins, other_ci_low, other_ci_high

def get_topic_bins(perf_metric, user_df, other_dfs, n_topics, ci=0.95, n_boot=501):  
    # Note: not using other_dfs anymore
    y_user = []
    y_other = []
    used_bins = []
    other_ci_low = []
    other_ci_high = []
    selected_topics = unique_topics_ids[1:(n_topics + 1)]
    
    for topic_id in selected_topics:
        cur_topic = topic_ids_to_topics[topic_id]
        metric_others = [metrics[get_metric_ind(perf_metric)] for metrics in perf_1000_topics[topic_id].values() if metrics[get_metric_ind(perf_metric)]]
        ci_low, ci_high = mne.stats.bootstrap_confidence_interval(np.array(metric_others), ci=ci, n_bootstraps=n_boot, stat_fun='median')
        metric_other = np.median(metric_others)
        
        cur_user_df = user_df[user_df["topic"] == cur_topic]
        y_true_user = cur_user_df.pred.to_numpy()  # user's label
        y_pred = cur_user_df.rating_avg.to_numpy()  # system's label (avg)
        
        if len(y_true_user) > 0:
            used_bins.append(cur_topic)
            metric_user = calc_metric_user(y_true_user, y_pred, perf_metric)
            y_user.append(metric_user)
            y_other.append(metric_other)
            other_ci_low.append(ci_low)
            other_ci_high.append(ci_high)
            
    return y_user, y_other, used_bins, other_ci_low, other_ci_high

def calc_metric_user(y_true_user, y_pred, perf_metric):
    if perf_metric == "MAE":
        metric_user = mean_absolute_error(y_true_user, y_pred)

    elif perf_metric == "MSE":
        metric_user = mean_squared_error(y_true_user, y_pred)

    elif perf_metric == "RMSE":            
        metric_user = mean_squared_error(y_true_user, y_pred, squared=False)
        
    elif perf_metric == "avg_diff":
        metric_user = np.mean(y_true_user - y_pred)
    
    return metric_user

def get_toxicity_category_bins(perf_metric, user_df, other_dfs, threshold=0.5, ci=0.95, n_boot=501):
    # Note: not using other_dfs anymore; threshold from pre-calculation is 0.5
    cat_cols = ["is_profane_frac", "is_threat_frac", "is_identity_attack_frac", "is_insult_frac", "is_sexual_harassment_frac"]
    cat_labels = ["Profanity", "Threats", "Identity Attacks", "Insults", "Sexual Harassment"]
    y_user = []
    y_other = []
    used_bins = []
    other_ci_low = []
    other_ci_high = []
    for i, cur_col_name in enumerate(cat_cols):
        metric_others = [metrics[get_metric_ind(perf_metric)] for metrics in perf_1000_tox_cat[cur_col_name].values() if metrics[get_metric_ind(perf_metric)]]
        ci_low, ci_high = mne.stats.bootstrap_confidence_interval(np.array(metric_others), ci=ci, n_bootstraps=n_boot, stat_fun='median')
        metric_other = np.median(metric_others)
        
        # Filter to rows where a comment received an average label >= the provided threshold for the category
        cur_user_df = user_df[user_df[cur_col_name] >= threshold]
        y_true_user = cur_user_df.pred.to_numpy()  # user's label
        y_pred = cur_user_df.rating_avg.to_numpy()  # system's label (avg)
        
        if len(y_true_user) > 0:
            used_bins.append(cat_labels[i])
            metric_user = calc_metric_user(y_true_user, y_pred, perf_metric)
            y_user.append(metric_user)
            y_other.append(metric_other)
            other_ci_low.append(ci_low)
            other_ci_high.append(ci_high)
    
    return y_user, y_other, used_bins, other_ci_low, other_ci_high

def plot_class_cond_results(preds_df, breakdown_axis, perf_metric, other_ids, sort_bars, n_topics, worker_id="A"):
    # Note: preds_df already has binned results
    # Prepare dfs
    user_df = preds_df[preds_df.user_id == worker_id].sort_values(by=["item_id"]).reset_index()
    other_dfs = [preds_df[preds_df.user_id == other_id].sort_values(by=["item_id"]).reset_index() for other_id in other_ids]
    
    if breakdown_axis == "toxicity_severity":
        y_user, y_other, used_bins, other_ci_low, other_ci_high = get_toxicity_severity_bins(perf_metric, user_df, other_dfs)
    elif breakdown_axis == "topic":
        y_user, y_other, used_bins, other_ci_low, other_ci_high = get_topic_bins(perf_metric, user_df, other_dfs, n_topics)
    elif breakdown_axis == "toxicity_category":
        y_user, y_other, used_bins, other_ci_low, other_ci_high = get_toxicity_category_bins(perf_metric, user_df, other_dfs)
    
    diffs = list(np.array(y_user) - np.array(y_other))
        
    # Generate bar chart
    data = pd.DataFrame({
        "metric_val": y_user + y_other,
        "Labeler": ["You" for _ in range(len(y_user))] + ["Other users" for _ in range(len(y_user))],
        "used_bins": used_bins + used_bins,
        "diffs": diffs + diffs,
        "lower_cis": y_user + other_ci_low,
        "upper_cis": y_user + other_ci_high,
    })

    color_domain = ['You', 'Other users']
    color_range = [YOUR_COLOR, OTHER_USERS_COLOR]
    
    base = alt.Chart()
    chart_title=f"{internal_to_readable[breakdown_axis]} Results"
    x_axis = alt.X("Labeler:O", sort=("You", "Other users"), title=None, axis=None)
    y_axis = alt.Y("metric_val:Q", title=internal_to_readable[perf_metric])
    if sort_bars:
        col_content = alt.Column("used_bins:O", sort=alt.EncodingSortField(field="diffs", op="mean", order='descending'))
    else:
        col_content = alt.Column("used_bins:O")

    if n_topics is not None and n_topics > 10:
        # Change to horizontal bar chart
        bar = base.mark_bar(lineBreak="_").encode(
            y=x_axis,
            x=y_axis,
            color=alt.Color("Labeler:O", scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[
                alt.Tooltip('Labeler:O', title='Labeler'),
                alt.Tooltip('metric_val:Q', title=perf_metric, format=".3f"),
            ]
        )
        error_bars = base.mark_errorbar().encode(
            y=x_axis,
            x = alt.X("lower_cis:Q", title=internal_to_readable[perf_metric]),
            x2 = alt.X2("upper_cis:Q", title=None),
            tooltip=[
              alt.Tooltip('lower_cis:Q', title='Lower CI', format=".3f"),
              alt.Tooltip('upper_cis:Q', title='Upper CI', format=".3f"),
            ]
        )
        combined = alt.layer(
            bar, error_bars, data=data
        ).facet(
            row=col_content
        ).properties(
            title=chart_title,
        ).interactive()
    else:
        bar = base.mark_bar(lineBreak="_").encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color("Labeler:O", scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[
                alt.Tooltip('Labeler:O', title='Labeler'),
                alt.Tooltip('metric_val:Q', title=perf_metric, format=".3f"),
            ]
        )
        error_bars = base.mark_errorbar().encode(
            x=x_axis,
            y = alt.Y("lower_cis:Q", title=internal_to_readable[perf_metric]),
            y2 = alt.Y2("upper_cis:Q", title=None),
            tooltip=[
              alt.Tooltip('lower_cis:Q', title='Lower CI', format=".3f"),
              alt.Tooltip('upper_cis:Q', title='Upper CI', format=".3f"),
            ]
        )
        combined = alt.layer(
            bar, error_bars, data=data
        ).facet(
            column=col_content
        ).properties(
            title=chart_title,
        ).interactive()

    return combined

# Generates the summary plot across all topics for the user
def show_overall_perf(variant, error_type, cur_user, threshold=TOXIC_THRESHOLD, breakdown_axis=None, topic_vis_method="median"):
    # Your perf (calculate using model and testset)
    breakdown_axis = readable_to_internal[breakdown_axis]
    
    if breakdown_axis is not None:
        with open(os.path.join(module_dir, f"data/preds_dfs/{variant}.pkl"), "rb") as f:
            preds_df = pickle.load(f)

        # Read from file
        chart_dir = "./data/charts"
        chart_file = os.path.join(chart_dir, f"{cur_user}_{variant}.pkl")
        if os.path.isfile(chart_file):
            with open(chart_file, "r") as f:
                topic_overview_plot_json = json.load(f)
        else:
            preds_df_mod = preds_df.merge(comments_grouped_full_topic_cat, on="item_id", how="left", suffixes=('_', '_avg'))
            if topic_vis_method == "median":  # Default
                preds_df_mod_grp = preds_df_mod.groupby(["topic_", "user_id"]).median()
            elif topic_vis_method == "mean":
                preds_df_mod_grp = preds_df_mod.groupby(["topic_", "user_id"]).mean()
            topic_overview_plot_json = plot_overall_vis(preds_df=preds_df_mod_grp, n_topics=200, threshold=threshold, error_type=error_type, cur_user=cur_user, cur_model=variant)

    return {
        "topic_overview_plot_json": json.loads(topic_overview_plot_json),
    }

########################################
# GET_CLUSTER_RESULTS utils
def get_overall_perf3(preds_df, perf_metric, other_ids, worker_id="A"):    
    # Prepare dataset to calculate performance
    # Note: true is user and pred is system
    y_true = preds_df[preds_df["user_id"] == worker_id].pred.to_numpy()
    y_pred_user = preds_df[preds_df["user_id"] == worker_id].rating_avg.to_numpy()
    
    y_true_others = y_pred_others = [preds_df[preds_df["user_id"] == other_id].pred.to_numpy() for other_id in other_ids]
    y_pred_others = [preds_df[preds_df["user_id"] == other_id].rating_avg.to_numpy() for other_id in other_ids]
    
    # Get performance for user's model and for other users
    if perf_metric == "MAE":
        user_perf = mean_absolute_error(y_true, y_pred_user)
        other_perfs = [mean_absolute_error(y_true_others[i], y_pred_others[i]) for i in range(len(y_true_others))]
    elif perf_metric == "MSE":
        user_perf = mean_squared_error(y_true, y_pred_user)
        other_perfs = [mean_squared_error(y_true_others[i], y_pred_others[i]) for i in range(len(y_true_others))]
    elif perf_metric == "RMSE":
        user_perf = mean_squared_error(y_true, y_pred_user, squared=False)
        other_perfs = [mean_squared_error(y_true_others[i], y_pred_others[i], squared=False) for i in range(len(y_true_others))]
    elif perf_metric == "avg_diff":
        user_perf = np.mean(y_true - y_pred_user)
        other_perfs = [np.mean(y_true_others[i] - y_pred_others[i]) for i in range(len(y_true_others))]
    
    other_perf = np.mean(other_perfs)  # average across all other users
    return user_perf, other_perf

def style_color_difference(row):
    full_opacity_diff = 3.
    pred_user_col = "Your predicted rating"
    pred_other_col = "Other users' predicted rating"
    pred_system_col = "Status-quo system rating"
    diff_user = row[pred_user_col] - row[pred_system_col]
    diff_other = row[pred_other_col] - row[pred_system_col]
    red = "234, 133, 125"
    green = "142, 205, 162"
    bkgd_user = green if diff_user < 0 else red  # red if more toxic; green if less toxic
    opac_user = min(abs(diff_user / full_opacity_diff), 1.)
    bkgd_other = green if diff_other < 0 else red  # red if more toxic; green if less toxic
    opac_other = min(abs(diff_other / full_opacity_diff), 1.)
    return ["", f"background-color: rgba({bkgd_user}, {opac_user});", f"background-color: rgba({bkgd_other}, {opac_other});", "", ""]

def display_examples_cluster(preds_df, other_ids, num_examples, sort_ascending, worker_id="A"):
    user_df = preds_df[preds_df.user_id == worker_id].sort_values(by=["item_id"]).reset_index()
    others_df = preds_df[preds_df.user_id == other_ids[0]]
    for i in range(1, len(other_ids)):
        others_df.append(preds_df[preds_df.user_id == other_ids[i]])
        others_df.groupby(["item_id"]).mean()
    others_df = others_df.sort_values(by=["item_id"]).reset_index()
    
    df = pd.merge(user_df, others_df, on="item_id", how="left", suffixes=('_user', '_other'))
    df["Comment"] = df["comment_user"]
    df["Your predicted rating"] = df["pred_user"]
    df["Other users' predicted rating"] = df["pred_other"]
    df["Status-quo system rating"] = df["rating_avg_user"]
    df["Status-quo system std dev"] = df["rating_stddev_user"]
    df = df[["Comment", "Your predicted rating", "Other users' predicted rating", "Status-quo system rating", "Status-quo system std dev"]]
    
    # Add styling
    df = df.sort_values(by=['Status-quo system std dev'], ascending=sort_ascending)
    n_to_sample = np.min([num_examples, len(df)])
    df = df.sample(n=n_to_sample).reset_index(drop=True)
    return df.style.apply(style_color_difference, axis=1).render()

def calc_odds_ratio(df, comparison_group, toxic_threshold=1.5, worker_id="A", debug=False, smoothing_factor=1):
    if comparison_group == "status_quo":
        other_pred_col = "rating_avg"
        # Get unique comments, but fetch average labeler rating
        num_toxic_other = len(df[(df.user_id == "A") & (df[other_pred_col] >= toxic_threshold)]) + smoothing_factor
        num_nontoxic_other = len(df[(df.user_id == "A") & (df[other_pred_col] < toxic_threshold)]) + smoothing_factor
    elif comparison_group == "other_users":
        other_pred_col = "pred"
        num_toxic_other = len(df[(df.user_id != "A") & (df[other_pred_col] >= toxic_threshold)]) + smoothing_factor
        num_nontoxic_other = len(df[(df.user_id != "A") & (df[other_pred_col] < toxic_threshold)]) + smoothing_factor
        
    num_toxic_user = len(df[(df.user_id == "A") & (df.pred >= toxic_threshold)]) + smoothing_factor
    num_nontoxic_user = len(df[(df.user_id == "A") & (df.pred < toxic_threshold)]) + smoothing_factor
    
    toxic_ratio = num_toxic_user / num_toxic_other
    nontoxic_ratio = num_nontoxic_user / num_nontoxic_other
    odds_ratio = toxic_ratio / nontoxic_ratio
    
    if debug:
        print(f"Odds ratio: {odds_ratio}")
        print(f"num_toxic_user: {num_toxic_user}, num_nontoxic_user: {num_nontoxic_user}")
        print(f"num_toxic_other: {num_toxic_other}, num_nontoxic_other: {num_nontoxic_other}")
    
    contingency_table = [[num_toxic_user, num_nontoxic_user], [num_toxic_other, num_nontoxic_other]]
    odds_ratio, p_val = stats.fisher_exact(contingency_table, alternative='two-sided')
    if debug:
        print(f"Odds ratio: {odds_ratio}, p={p_val}")

    return odds_ratio

# Neighbor search
def get_match(comment_inds, K=20, threshold=None, debug=False):
    match_ids = []
    rows = []
    for i in comment_inds:
        if debug:
            print(f"\nComment: {comments[i]}")
        query_embedding = model.encode(comments[i], convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, embeddings, score_function=util.cos_sim, top_k=K)
        # print(hits[0])
        for hit in hits[0]:
            c_id = hit['corpus_id']
            score = np.round(hit['score'], 3)
            if threshold is None or score > threshold:
                match_ids.append(c_id)
                if debug:
                    print(f"\t(ID={c_id}, Score={score}): {comments[c_id]}")
                rows.append([c_id, score, comments[c_id]])
    
    df = pd.DataFrame(rows, columns=["id", "score", "comment"])
    return match_ids

def display_examples_auto_cluster(preds_df, cluster, other_ids, perf_metric, sort_ascending=True, worker_id="A", num_examples=10):
    # Overall performance
    topic_df = preds_df
    topic_df = topic_df[topic_df["topic"] == cluster]
    user_perf, other_perf = get_overall_perf3(topic_df, perf_metric, other_ids)
    
    user_direction = "LOWER" if user_perf < 0 else "HIGHER"
    other_direction = "LOWER" if other_perf < 0 else "HIGHER"
    print(f"Your ratings are on average {np.round(abs(user_perf), 3)} {user_direction} than the existing system for this cluster")
    print(f"Others' ratings (based on {len(other_ids)} users) are on average {np.round(abs(other_perf), 3)} {other_direction} than the existing system for this cluster")
        
    # Display example comments
    df = display_examples_cluster(preds_df, other_ids, num_examples, sort_ascending)
    return df

    
# function to get results for a new provided cluster
def display_examples_manual_cluster(preds_df, cluster_comments, other_ids, perf_metric, sort_ascending=True, worker_id="A"):
    # Overall performance
    cluster_df = preds_df[preds_df["comment"].isin(cluster_comments)]
    user_perf, other_perf = get_overall_perf3(cluster_df, perf_metric, other_ids)
    
    user_direction = "LOWER" if user_perf < 0 else "HIGHER"
    other_direction = "LOWER" if other_perf < 0 else "HIGHER"
    print(f"Your ratings are on average {np.round(abs(user_perf), 3)} {user_direction} than the existing system for this cluster")
    print(f"Others' ratings (based on {len(other_ids)} users) are on average {np.round(abs(other_perf), 3)} {other_direction} than the existing system for this cluster")
        
    user_df = preds_df[preds_df.user_id == worker_id].sort_values(by=["item_id"]).reset_index()
    others_df = preds_df[preds_df.user_id == other_ids[0]]
    for i in range(1, len(other_ids)):
        others_df.append(preds_df[preds_df.user_id == other_ids[i]])
        others_df.groupby(["item_id"]).mean()
    others_df = others_df.sort_values(by=["item_id"]).reset_index()
    
    # Get cluster_comments
    user_df = user_df[user_df["comment"].isin(cluster_comments)]
    others_df = others_df[others_df["comment"].isin(cluster_comments)]
    
    df = pd.merge(user_df, others_df, on="item_id", how="left", suffixes=('_user', '_other'))
    df["pred_system"] = df["rating_avg_user"]
    df["pred_system_stddev"] = df["rating_stddev_user"]
    df = df[["item_id", "comment_user", "pred_user", "pred_other", "pred_system", "pred_system_stddev"]]
    
    # Add styling
    df = df.sort_values(by=['pred_system_stddev'], ascending=sort_ascending)
    df = df.style.apply(style_color_difference, axis=1).render()
    return df

########################################
# GET_LABELING utils
def create_example_sets(comments_df, n_label_per_bin, score_bins, keyword=None, topic=None):
    # Restrict to the keyword, if provided
    df = comments_df.copy()
    if keyword != None:
        df = df[df["comment"].str.contains(keyword)]    
    
    if topic != None:
        df = df[df["topic"] == topic]  
    
    # Try to choose n values from each provided score bin
    ex_to_label = []
    bin_names = []
    bin_label_counts = []
    for i, score_bin in enumerate(score_bins):
        min_score, max_score = score_bin
        cur_df = df[(df["rating"] >= min_score) & (df["rating"] < max_score) & (df["item_id"].isin(train_df_ids))]
        # sample rows for label
        comment_ids = cur_df.item_id.tolist()
        cur_n_label_per_bin = n_label_per_bin[i]
        cap = min(len(comment_ids), (cur_n_label_per_bin))
        to_label = np.random.choice(comment_ids, cap, replace=False)
        ex_to_label.extend(to_label)
        bin_names.append(f"[{min_score}, {max_score})")
        bin_label_counts.append(len(to_label))
    
    return ex_to_label

def get_grp_model_labels(comments_df, n_label_per_bin, score_bins, grp_ids):
    df = comments_df.copy()

    train_df_grp = train_df[train_df["user_id"].isin(grp_ids)]
    train_df_grp_avg = train_df_grp.groupby(by=["item_id"]).median().reset_index()
    train_df_grp_avg_ids = train_df_grp_avg["item_id"].tolist()

    ex_to_label = [] # IDs of comments to use for group model training
    for i, score_bin in enumerate(score_bins):
        min_score, max_score = score_bin
        # get eligible comments to sample
        cur_df = df[(df["rating"] >= min_score) & (df["rating"] < max_score) & (df["item_id"].isin(train_df_grp_avg_ids))]
        comment_ids = cur_df.item_id.unique().tolist()
        # sample comments
        cur_n_label_per_bin = n_label_per_bin[i]
        cap = min(len(comment_ids), (cur_n_label_per_bin))
        to_label = np.random.choice(comment_ids, cap, replace=False)
        ex_to_label.extend((to_label))
    
    train_df_grp_avg = train_df_grp_avg[train_df_grp_avg["item_id"].isin(ex_to_label)]

    ratings_grp = {ids_to_comments[int(r["item_id"])]: r["rating"] for _, r in train_df_grp_avg.iterrows()}

    return ratings_grp  

########################################
# GET_PERSONALIZED_MODEL utils
def fetch_existing_data(model_name, last_label_i):
    # Check if we have cached model performance
    perf_dir = f"./data/perf/{model_name}"
    label_dir = f"./data/labels/{model_name}"
    if os.path.isdir(os.path.join(module_dir, perf_dir)):
        # Fetch cached results
        last_i = len([name for name in os.listdir(os.path.join(module_dir, perf_dir)) if os.path.isfile(os.path.join(module_dir, perf_dir, name))])
        with open(os.path.join(module_dir, perf_dir, f"{last_i}.pkl"), "rb") as f:
            mae, mse, rmse, avg_diff = pickle.load(f)
    else:
        # Fetch results from trained model
        with open(os.path.join(module_dir, f"./data/trained_models/{model_name}.pkl"), "rb") as f:
            cur_model = pickle.load(f)
            mae, mse, rmse, avg_diff = users_perf(cur_model)
        # Cache results
        os.mkdir(os.path.join(module_dir, perf_dir))
        with open(os.path.join(module_dir, perf_dir, "1.pkl"), "wb") as f:
            pickle.dump((mae, mse, rmse, avg_diff), f)
    
    # Fetch previous user-provided labels
    ratings_prev = None
    if last_label_i > 0:
        with open(os.path.join(module_dir, label_dir, f"{last_i}.pkl"), "rb") as f:
            ratings_prev = pickle.load(f)
    return mae, mse, rmse, avg_diff, ratings_prev

def train_updated_model(model_name, last_label_i, ratings, user, top_n=20, topic=None):
    # Check if there is previously-labeled data; if so, combine it with this data
    perf_dir = f"./data/perf/{model_name}"
    label_dir = f"./data/labels/{model_name}"
    labeled_df = format_labeled_data(ratings) # Treat ratings as full batch of all ratings
    ratings_prev = None

    # Filter out rows with "unsure" (-1)
    labeled_df = labeled_df[labeled_df["rating"] != -1]

    # Filter to top N for user study
    if topic is None:
        # labeled_df = labeled_df.head(top_n)
        labeled_df = labeled_df.tail(top_n)
    else:
        # For topic tuning, need to fetch old labels
        if (last_label_i > 0):
            # Concatenate previous set of labels with this new batch of labels
            with open(os.path.join(module_dir, label_dir, f"{last_label_i}.pkl"), "rb") as f:
                ratings_prev = pickle.load(f)
                labeled_df_prev = format_labeled_data(ratings_prev)
                labeled_df_prev = labeled_df_prev[labeled_df_prev["rating"] != -1]
                ratings.update(ratings_prev) # append old ratings to ratings
                labeled_df = pd.concat([labeled_df_prev, labeled_df])

    print("len ratings for training:", len(labeled_df))

    cur_model, perf, _, _ = train_user_model(ratings_df=labeled_df)
    
    user_perf_metrics[model_name] = users_perf(cur_model)

    mae, mse, rmse, avg_diff = user_perf_metrics[model_name]

    cur_preds_df = get_preds_df(cur_model, ["A"], sys_eval_df=ratings_df_full)  # Just get results for user

    # Save this batch of labels
    with open(os.path.join(module_dir, label_dir, f"{last_label_i + 1}.pkl"), "wb") as f:
        pickle.dump(ratings, f)

    # Save model results
    with open(os.path.join(module_dir, f"./data/preds_dfs/{model_name}.pkl"), "wb") as f:
        pickle.dump(cur_preds_df, f)

    if model_name not in all_model_names:
        all_model_names.append(model_name)
    with open(os.path.join(module_dir, "./data/all_model_names.pkl"), "wb") as f:
        pickle.dump(all_model_names, f)
    
    # Handle user
    if user not in users_to_models:
        users_to_models[user] = []  # New user
    if model_name not in users_to_models[user]:
        users_to_models[user].append(model_name)  # New model
        with open(f"./data/users_to_models.pkl", "wb") as f:
            pickle.dump(users_to_models, f)

    with open(os.path.join(module_dir, "./data/user_perf_metrics.pkl"), "wb") as f:
        pickle.dump(user_perf_metrics, f)
    with open(os.path.join(module_dir, f"./data/trained_models/{model_name}.pkl"), "wb") as f:
        pickle.dump(cur_model, f)
    
    # Cache performance results
    if not os.path.isdir(os.path.join(module_dir, perf_dir)):
        os.mkdir(os.path.join(module_dir, perf_dir))
    last_perf_i = len([name for name in os.listdir(os.path.join(module_dir, perf_dir)) if os.path.isfile(os.path.join(module_dir, perf_dir, name))])
    with open(os.path.join(module_dir, perf_dir, f"{last_perf_i + 1}.pkl"), "wb") as f:
        pickle.dump((mae, mse, rmse, avg_diff), f)

    ratings_prev = ratings
    return mae, mse, rmse, avg_diff, ratings_prev

def format_labeled_data(ratings, worker_id="A", debug=False):    
    all_rows = []
    for comment, rating in ratings.items():
        comment_id = comments_to_ids[comment]
        row = [worker_id, comment_id, int(rating)]
        all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=["user_id", "item_id", "rating"])
    return df

def users_perf(model, sys_eval_df=sys_eval_df, avg_ratings_df=comments_grouped_full_topic_cat, worker_id="A"):
    # Load the full empty dataset
    sys_eval_comment_ids = sys_eval_df.item_id.unique().tolist()
    empty_ratings_rows = [[worker_id, c_id, 0] for c_id in sys_eval_comment_ids]
    empty_ratings_df = pd.DataFrame(empty_ratings_rows, columns=["user_id", "item_id", "rating"])

    # Compute predictions for full dataset
    reader = Reader(rating_scale=(0, 4))
    eval_set_data = Dataset.load_from_df(empty_ratings_df, reader)
    _, testset = train_test_split(eval_set_data, test_size=1.)
    predictions = model.test(testset)

    df = empty_ratings_df # user_id, item_id, rating
    user_item_preds = get_predictions_by_user_and_item(predictions)
    df["pred"] = df.apply(lambda row: user_item_preds[(row.user_id, row.item_id)] if (row.user_id, row.item_id) in user_item_preds else np.nan, axis=1)

    df = df.merge(avg_ratings_df, on="item_id", how="left", suffixes=('_', '_avg'))
    df.dropna(subset = ["pred"], inplace=True)
    df["rating_"] = df.rating_.astype("int32")

    perf_metrics = get_overall_perf(df, "A") # mae, mse, rmse, avg_diff  
    return perf_metrics

def get_overall_perf(preds_df, user_id):    
    # Prepare dataset to calculate performance
    y_pred = preds_df[preds_df["user_id"] == user_id].rating_avg.to_numpy() # Assume system is just average of true labels
    y_true = preds_df[preds_df["user_id"] == user_id].pred.to_numpy()
    
    # Get performance for user's model
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    avg_diff = np.mean(y_true - y_pred)
    
    return mae, mse, rmse, avg_diff

def get_predictions_by_user_and_item(predictions):
    user_item_preds = {}
    for uid, iid, true_r, est, _ in predictions:
        user_item_preds[(uid, iid)] = est
    return user_item_preds

# Pre-computes predictions for the provided model and specified users on the system-eval dataset
# - model: trained model
# - user_ids: list of user IDs to compute predictions for
# - avg_ratings_df: dataframe of average ratings for each comment (pre-computed)
# - sys_eval_df: dataframe of system eval labels (pre-computed)
def get_preds_df(model, user_ids, avg_ratings_df=comments_grouped_full_topic_cat, sys_eval_df=sys_eval_df, bins=BINS):
    # Prep dataframe for all predictions we'd like to request
    start = time.time()
    sys_eval_comment_ids = sys_eval_df.item_id.unique().tolist()

    empty_ratings_rows = []
    for user_id in user_ids:
        empty_ratings_rows.extend([[user_id, c_id, 0] for c_id in sys_eval_comment_ids])
    empty_ratings_df = pd.DataFrame(empty_ratings_rows, columns=["user_id", "item_id", "rating"])
    print("setup", time.time() - start)
    
    # Evaluate model to get predictions
    start = time.time() 
    reader = Reader(rating_scale=(0, 4))
    eval_set_data = Dataset.load_from_df(empty_ratings_df, reader)
    _, testset = train_test_split(eval_set_data, test_size=1.)
    predictions = model.test(testset)
    print("train_test_split", time.time() - start)
    
    # Update dataframe with predictions
    start = time.time()
    df = empty_ratings_df.copy() # user_id, item_id, rating
    user_item_preds = get_predictions_by_user_and_item(predictions)
    df["pred"] = df.apply(lambda row: user_item_preds[(row.user_id, row.item_id)] if (row.user_id, row.item_id) in user_item_preds else np.nan, axis=1)
    df = df.merge(avg_ratings_df, on="item_id", how="left", suffixes=('_', '_avg'))
    df.dropna(subset = ["pred"], inplace=True)
    df["rating_"] = df.rating_.astype("int32")
    
    # Get binned predictions (based on user prediction)
    df["prediction_bin"], out_bins = pd.cut(df["pred"], bins, labels=False, retbins=True)
    df = df.sort_values(by=["item_id"])

    return df

# Given the full set of ratings, trains the specified model type and evaluates on the model eval set
# - ratings_df: dataframe of all ratings
# - train_df: dataframe of training labels
# - model_eval_df: dataframe of model eval labels (validation set)
# - train_frac: fraction of ratings to use for training
def train_user_model(ratings_df, train_df=train_df, model_eval_df=model_eval_df, train_frac=0.75, model_type="SVD", sim_type=None, user_based=True):
    # Sample from shuffled labeled dataframe and add batch to train set; specified set size to model_eval set
    labeled = ratings_df.sample(frac=1)  # Shuffle the data
    batch_size = math.floor(len(labeled) * train_frac)
    labeled_train = labeled[:batch_size]
    labeled_model_eval = labeled[batch_size:]
    
    train_df_ext = train_df.append(labeled_train)
    model_eval_df_ext = model_eval_df.append(labeled_model_eval)
    
    # Train model and show model eval set results
    model, perf = train_model(train_df_ext, model_eval_df_ext, model_type=model_type, sim_type=sim_type, user_based=user_based)
    
    return model, perf, labeled_train, labeled_model_eval

# Given a set of labels split into training and validation (model_eval), trains the specified model type on the training labels and evaluates on the model_eval labels
# - train_df: dataframe of training labels
# - model_eval_df: dataframe of model eval labels (validation set)
# - model_type: type of model to train
def train_model(train_df, model_eval_df, model_type="SVD", sim_type=None, user_based=True):
    # Train model
    reader = Reader(rating_scale=(0, 4))
    train_data = Dataset.load_from_df(train_df, reader)
    model_eval_data = Dataset.load_from_df(model_eval_df, reader)
    
    train_set = train_data.build_full_trainset()
    _, model_eval_set = train_test_split(model_eval_data, test_size=1.)

    sim_options = {
        "name": sim_type,
        "user_based": user_based, # compute similarity between users or items
    }
    if model_type == "SVD":
        algo = SVD()  # SVD doesn't have similarity metric
    elif model_type == "KNNBasic":
        algo = KNNBasic(sim_options=sim_options)
    elif model_type == "KNNWithMeans":
        algo = KNNWithMeans(sim_options=sim_options)
    elif model_type == "KNNWithZScore":
        algo = KNNWithZScore(sim_options=sim_options)
    algo.fit(train_set)
    
    predictions = algo.test(model_eval_set)
    rmse = accuracy.rmse(predictions)
    fcp = accuracy.fcp(predictions)
    mae = accuracy.mae(predictions)
    mse = accuracy.mse(predictions)
    
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, FCP: {fcp}")
    perf = [mae, mse, rmse, fcp]
    
    return algo, perf

def plot_train_perf_results2(model_name):
    # Open labels
    label_dir = f"./data/labels/{model_name}"
    n_label_files = len([name for name in os.listdir(os.path.join(module_dir, label_dir)) if os.path.isfile(os.path.join(module_dir, label_dir, name))])
    
    all_rows = []
    with open(os.path.join(module_dir, label_dir, f"{n_label_files}.pkl"), "rb") as f:
        ratings = pickle.load(f)

        labeled_df = format_labeled_data(ratings)
        labeled_df = labeled_df[labeled_df["rating"] != -1]

        # Iterate through batches of 5 labels
        n_batches = int(np.ceil(len(labeled_df) / 5.))
        for i in range(n_batches):
            start = time.time()
            n_to_sample = np.min([5 * (i + 1), len(labeled_df)])
            cur_model, _, _, _ = train_user_model(ratings_df=labeled_df.head(n_to_sample))
            mae, mse, rmse, avg_diff = users_perf(cur_model)
            all_rows.append([n_to_sample, mae, "MAE"])
            print(f"iter {i}: {time.time() - start}")
        
        print("all_rows", all_rows)
        
        df = pd.DataFrame(all_rows, columns=["n_to_sample", "perf", "metric"])
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("n_to_sample:Q", title="Number of Comments Labeled"),
            y="perf",
            color="metric",
            tooltip=[
                alt.Tooltip('n_to_sample:Q', title="Number of Comments Labeled"),
                alt.Tooltip('metric:N', title="Metric"),
                alt.Tooltip('perf:Q', title="Metric Value", format=".3f"),
            ],
        ).properties(
            title=f"Performance over number of examples: {model_name}",
            width=500,
        )
        return chart

def plot_train_perf_results(model_name, mae):
    perf_dir = f"./data/perf/{model_name}"
    n_perf_files = len([name for name in os.listdir(os.path.join(module_dir, perf_dir)) if os.path.isfile(os.path.join(module_dir, perf_dir, name))])

    all_rows = []
    for i in range(1, n_perf_files + 1):
        with open(os.path.join(module_dir, perf_dir, f"{i}.pkl"), "rb") as f:
            mae, mse, rmse, avg_diff = pickle.load(f)
            all_rows.append([i, mae, "Your MAE"])
        
    df = pd.DataFrame(all_rows, columns=["version", "perf", "metric"])
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="version:O",
        y="perf",
        color=alt.Color("metric", title="Performance metric"),
        tooltip=[
            alt.Tooltip('version:O', title='Version'),
            alt.Tooltip('metric:N', title="Metric"),
            alt.Tooltip('perf:Q', title="Metric Value", format=".3f"),
        ],
    ).properties(
        title=f"Performance over model versions: {model_name}",
        width=500,
    )

    PCT_50 = 0.591
    PCT_75 = 0.662
    PCT_90 = 0.869

    plot_dim_width = 500
    domain_min = 0.0
    domain_max = 1.0
    bkgd = alt.Chart(pd.DataFrame({
        "start": [PCT_90, PCT_75, domain_min],
        "stop": [domain_max, PCT_90, PCT_75],
        "bkgd": ["Needs improvement (< top 90%)", "Okay (top 90%)", "Good (top 75%)"],
    })).mark_rect(opacity=0.2).encode(
        y=alt.Y("start:Q", scale=alt.Scale(domain=[0, domain_max])),
        y2=alt.Y2("stop:Q"),
        x=alt.value(0),
        x2=alt.value(plot_dim_width),
        color=alt.Color("bkgd:O", scale=alt.Scale(
            domain=["Needs improvement (< top 90%)", "Okay (top 90%)", "Good (top 75%)"], 
            range=["red", "yellow", "green"]),
            title="How good is your MAE?"
        )
    )

    plot = (bkgd + chart).properties(width=plot_dim_width).resolve_scale(color='independent')
    mae_status = None
    if mae < PCT_75:
        mae_status = "Your MAE is in the <b>Good</b> range, which means that it's in the top 75% of scores compared to other users. Your model looks good to go."
    elif mae < PCT_90:
        mae_status = "Your MAE is in the <b>Okay</b> range, which means that it's in the top 90% of scores compared to other users. Your model can be used, but you can provide additional labels to improve it."
    else:
        mae_status = "Your MAE is in the <b>Needs improvement</b> range, which means that it's in below the top 95% of scores compared to other users. Your model may need additional labels to improve."
    return plot, mae_status

########################################
# New visualizations
# Constants
VIS_BINS = np.round(np.arange(0, 4.01, 0.05), 3)
VIS_BINS_LABELS = [np.round(np.mean([x, y]), 3) for x, y in zip(VIS_BINS[:-1], VIS_BINS[1:])]

def get_key(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return "System agrees: Non-toxic"
    elif sys > threshold and user > threshold:
        return "System agrees: Toxic"
    else:
        if abs(sys - threshold) > 1.5:
            return "System differs: Error > 1.5"
        elif abs(sys - threshold) > 1.0:
            return "System differs: Error > 1.0"
        elif abs(sys - threshold) > 0.5:
            return "System differs: Error > 0.5"
        else:
            return "System differs: Error <=0.5"

def get_key_no_model(sys, threshold):
    if sys <= threshold:
        return "System says: Non-toxic"
    else:
        return "System says: Toxic"

def get_user_color(user, threshold):
    if user <= threshold:
        return "#FFF"  # white
    else:
        return "#808080"  # grey

def get_system_color(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return "#FFF"  # white
    elif sys > threshold and user > threshold:
        return "#808080"  # grey
    else:
        if abs(sys - threshold) > 1.5:
            return "#d62728" # red
        elif abs(sys - threshold) > 1.0:
            return "#ff7a5c" # med red
        elif abs(sys - threshold) > 0.5:
            return "#ffa894" # light red
        else:
            return "#ffd1c7" # very light red

def get_error_type(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return "No error (agree non-toxic)"
    elif sys > threshold and user > threshold:
        return "No error (agree toxic)"
    elif sys <= threshold and user > threshold:
        return "System may be under-sensitive"
    elif sys > threshold and user <= threshold:
        return "System may be over-sensitive"

def get_error_type_radio(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return "Show errors and non-errors"
    elif sys > threshold and user > threshold:
        return "Show errors and non-errors"
    elif sys <= threshold and user > threshold:
        return "System is under-sensitive"
    elif sys > threshold and user <= threshold:
        return "System is over-sensitive"

def get_error_magnitude(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return 0  # no classification error
    elif sys > threshold and user > threshold:
        return 0  # no classification error
    elif sys <= threshold and user > threshold:
        return abs(sys - user)
    elif sys > threshold and user <= threshold:
        return abs(sys - user)

def get_error_size(sys, user, threshold):
    if sys <= threshold and user <= threshold:
        return 0  # no classification error
    elif sys > threshold and user > threshold:
        return 0  # no classification error
    elif sys <= threshold and user > threshold:
        return sys - user
    elif sys > threshold and user <= threshold:
        return sys - user

def get_decision(rating, threshold):
    if rating <= threshold:
        return "Non-toxic"
    else:
        return "Toxic"

def get_category(row, threshold=0.3):
    k_to_category = {
        "is_profane_frac_": "Profanity", 
        "is_threat_frac_": "Threat", 
        "is_identity_attack_frac_": "Identity Attack", 
        "is_insult_frac_": "Insult", 
        "is_sexual_harassment_frac_": "Sexual Harassment",
    }
    categories = []
    for k in ["is_profane_frac_", "is_threat_frac_", "is_identity_attack_frac_", "is_insult_frac_", "is_sexual_harassment_frac_"]:
        if row[k] > threshold:
            categories.append(k_to_category[k])
    
    if len(categories) > 0:
        return ", ".join(categories)
    else:
        return ""

def get_comment_url(row):
    return f"#{row['item_id']}/#comment"

def get_topic_url(row):
    return f"#{row['topic_']}/#topic"

# Plots overall results histogram (each block is a topic)
def plot_overall_vis(preds_df, error_type, cur_user, cur_model, n_topics=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, bin_step=0.05):
    df = preds_df.copy().reset_index()
    
    if n_topics is not None:
        df = df[df["topic_id_"] < n_topics]
    
    df["vis_pred_bin"], out_bins = pd.cut(df["pred"], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == "A"].sort_values(by=["item_id"]).reset_index()
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df["rating"].tolist()]
    df["threshold"] = [threshold for r in df["rating"].tolist()]
    df["key"] = [get_key(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
    df["url"] = df.apply(lambda row: get_topic_url(row), axis=1)
    
    # Plot sizing
    domain_min = 0
    domain_max = 4

    plot_dim_height = 500
    plot_dim_width = 750
    max_items = np.max(df["vis_pred_bin"].value_counts().tolist())
    mark_size = np.round(plot_dim_height / max_items) * 8
    if mark_size > 75:
        mark_size = 75
        plot_dim_height = 13 * max_items

    # Main chart
    chart = alt.Chart(df).mark_square(opacity=0.8, size=mark_size, stroke="grey", strokeWidth=0.5).transform_window(
        groupby=['vis_pred_bin'],
        sort=[{'field': 'rating'}],
        id='row_number()',
        ignorePeers=True,
    ).encode(
        x=alt.X('vis_pred_bin:Q', title="Our prediction of your rating", scale=alt.Scale(domain=(domain_min, domain_max))),
        y=alt.Y('id:O', title="Topics (ordered by System toxicity rating)", axis=alt.Axis(values=list(range(0, max_items, 5))), sort='descending'),
        color = alt.Color("key:O", scale=alt.Scale(
            domain=["System agrees: Non-toxic", "System agrees: Toxic", "System differs: Error > 1.5", "System differs: Error > 1.0", "System differs: Error > 0.5", "System differs: Error <=0.5"], 
            range=["white", "#cbcbcb", "red", "#ff7a5c", "#ffa894", "#ffd1c7"]),
            title="System rating (box color)"
        ),
        href="url:N",
        tooltip = [
            alt.Tooltip("topic_:N", title="Topic"),
            alt.Tooltip("system_label:N", title="System label"),
            alt.Tooltip("rating:Q", title="System rating", format=".2f"),
            alt.Tooltip("pred:Q", title="Your rating", format=".2f")
        ]
    )

    # Filter to specified error type
    if error_type == "System is under-sensitive":
        # FN: system rates non-toxic, but user rates toxic
        chart = chart.transform_filter(
            alt.FieldGTPredicate(field="pred", gt=threshold)
        )
    elif error_type == "System is over-sensitive":
        # FP: system rates toxic, but user rates non-toxic
        chart = chart.transform_filter(
            alt.FieldLTEPredicate(field="pred", lte=threshold)
        )
    
    # Threshold line
    rule = alt.Chart(pd.DataFrame({
        "threshold": [threshold],
        "System threshold": [f"Threshold = {threshold}"]
    })).mark_rule().encode(
        x=alt.X("mean(threshold):Q", scale=alt.Scale(domain=(domain_min, domain_max)), title=""),
        color=alt.Color("System threshold:N", scale=alt.Scale(domain=[f"Threshold = {threshold}"], range=["grey"])),
        size=alt.value(2),
    )
    
    # Plot region annotations
    nontoxic_x = (domain_min + threshold) / 2.
    toxic_x = (domain_max + threshold) / 2.
    annotation = alt.Chart(pd.DataFrame({
        "annotation_text": ["Non-toxic", "Toxic"],
        "x": [nontoxic_x, toxic_x],
        "y": [max_items, max_items],
    })).mark_text(
        align="center",
        baseline="middle",
        fontSize=16,
        dy=10,
        color="grey"
    ).encode(
        x=alt.X("x", title=""),
        y=alt.Y("y", title="", axis=None),
        text="annotation_text"
    )
    
    # Plot region background colors
    bkgd = alt.Chart(pd.DataFrame({
        "start": [domain_min, threshold],
        "stop": [threshold, domain_max],
        "bkgd": ["Non-toxic (L side)", "Toxic (R side)"],
    })).mark_rect(opacity=1.0, stroke="grey", strokeWidth=0.25).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        x2=alt.X2("stop:Q"),
        y=alt.value(0),
        y2=alt.value(plot_dim_height),
        color=alt.Color("bkgd:O", scale=alt.Scale(
            domain=["Non-toxic (L side)", "Toxic (R side)"], 
            range=["white", "#cbcbcb"]),
            title="Your rating (background color)"
        )
    )
    
    plot = (bkgd + annotation + chart + rule).properties(height=(plot_dim_height), width=plot_dim_width).resolve_scale(color='independent').to_json()

    # Save to file
    chart_dir = "./data/charts"
    chart_file = os.path.join(chart_dir, f"{cur_user}_{cur_model}.pkl")
    with open(chart_file, "w") as f:
        json.dump(plot, f)

    return plot

# Plots cluster results histogram (each block is a comment), but *without* a model 
# as a point of reference (in contrast to plot_overall_vis_cluster)
def plot_overall_vis_cluster_no_model(preds_df, n_comments=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, bin_step=0.05):
    df = preds_df.copy().reset_index()
    
    df["vis_pred_bin"], out_bins = pd.cut(df["rating"], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == "A"].sort_values(by=["rating"]).reset_index()
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df["rating"].tolist()]
    df["key"] = [get_key_no_model(sys, threshold) for sys in df["rating"].tolist()]
    df["category"] = df.apply(lambda row: get_category(row), axis=1)
    df["url"] = df.apply(lambda row: get_comment_url(row), axis=1)
    
    if n_comments is not None:
        n_to_sample = np.min([n_comments, len(df)])
        df = df.sample(n=n_to_sample)
    
    # Plot sizing
    domain_min = 0
    domain_max = 4
    plot_dim_height = 500
    plot_dim_width = 750
    max_items = np.max(df["vis_pred_bin"].value_counts().tolist())
    mark_size = np.round(plot_dim_height / max_items) * 8
    if mark_size > 75:
        mark_size = 75
        plot_dim_height = 13 * max_items
    
    # Main chart
    chart = alt.Chart(df).mark_square(opacity=0.8, size=mark_size, stroke="grey", strokeWidth=0.25).transform_window(
        groupby=['vis_pred_bin'],
        sort=[{'field': 'rating'}],
        id='row_number()',
        ignorePeers=True
    ).encode(
        x=alt.X('vis_pred_bin:Q', title="System toxicity rating", scale=alt.Scale(domain=(domain_min, domain_max))),
        y=alt.Y('id:O', title="Comments (ordered by System toxicity rating)", axis=alt.Axis(values=list(range(0, max_items, 5))), sort='descending'),
        color = alt.Color("key:O", scale=alt.Scale(
            domain=["System says: Non-toxic", "System says: Toxic"], 
            range=["white", "#cbcbcb"]),
            title="System rating",
            legend=None,
        ),
        href="url:N",
        tooltip = [
            alt.Tooltip("comment_:N", title="comment"),
            alt.Tooltip("rating:Q", title="System rating", format=".2f"),
        ]
    )
    
    # Threshold line
    rule = alt.Chart(pd.DataFrame({
        "threshold": [threshold],
    })).mark_rule(color='grey').encode(
        x=alt.X("mean(threshold):Q", scale=alt.Scale(domain=[domain_min, domain_max]), title=""),
        size=alt.value(2),
    )
    
    # Plot region annotations
    nontoxic_x = (domain_min + threshold) / 2.
    toxic_x = (domain_max + threshold) / 2.
    annotation = alt.Chart(pd.DataFrame({
        "annotation_text": ["Non-toxic", "Toxic"],
        "x": [nontoxic_x, toxic_x],
        "y": [max_items, max_items],
    })).mark_text(
        align="center",
        baseline="middle",
        fontSize=16,
        dy=10,
        color="grey"
    ).encode(
        x=alt.X("x", title=""),
        y=alt.Y("y", title="", axis=None),
        text="annotation_text"
    )
    
    # Plot region background colors
    bkgd = alt.Chart(pd.DataFrame({
        "start": [domain_min, threshold],
        "stop": [threshold, domain_max],
        "bkgd": ["Non-toxic", "Toxic"],
    })).mark_rect(opacity=1.0, stroke="grey", strokeWidth=0.25).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        x2=alt.X2("stop:Q"),
        y=alt.value(0),
        y2=alt.value(plot_dim_height),
        color=alt.Color("bkgd:O", scale=alt.Scale(
            domain=["Non-toxic", "Toxic"], 
            range=["white", "#cbcbcb"]),
            title="System rating"
        )
    )
    
    final_plot = (bkgd + annotation + chart + rule).properties(height=(plot_dim_height), width=plot_dim_width).resolve_scale(color='independent').to_json()

    return final_plot, df

# Plots cluster results histogram (each block is a comment) *with* a model as a point of reference
def plot_overall_vis_cluster(preds_df, error_type, n_comments=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, bin_step=0.05):
    df = preds_df.copy().reset_index(drop=True)
    
    df["vis_pred_bin"], out_bins = pd.cut(df["pred"], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == "A"].sort_values(by=["rating"]).reset_index(drop=True)
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df["rating"].tolist()]
    df["key"] = [get_key(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
    df["category"] = df.apply(lambda row: get_category(row), axis=1)
    df["url"] = df.apply(lambda row: get_comment_url(row), axis=1)
    
    if n_comments is not None:
        n_to_sample = np.min([n_comments, len(df)])
        df = df.sample(n=n_to_sample)
    
    # Plot sizing
    domain_min = 0
    domain_max = 4
    plot_dim_height = 500
    plot_dim_width = 750
    max_items = np.max(df["vis_pred_bin"].value_counts().tolist())
    mark_size = np.round(plot_dim_height / max_items) * 8
    if mark_size > 75:
        mark_size = 75
        plot_dim_height = 13 * max_items
    
    # Main chart
    chart = alt.Chart(df).mark_square(opacity=0.8, size=mark_size, stroke="grey", strokeWidth=0.25).transform_window(
        groupby=['vis_pred_bin'],
        sort=[{'field': 'rating'}],
        id='row_number()',
        ignorePeers=True
    ).encode(
        x=alt.X('vis_pred_bin:Q', title="Our prediction of your rating", scale=alt.Scale(domain=(domain_min, domain_max))),
        y=alt.Y('id:O', title="Comments (ordered by System toxicity rating)", axis=alt.Axis(values=list(range(0, max_items, 5))), sort='descending'),
        color = alt.Color("key:O", scale=alt.Scale(
            domain=["System agrees: Non-toxic", "System agrees: Toxic", "System differs: Error > 1.5", "System differs: Error > 1.0", "System differs: Error > 0.5", "System differs: Error <=0.5"], 
            range=["white", "#cbcbcb", "red", "#ff7a5c", "#ffa894", "#ffd1c7"]),
            title="System rating (box color)"
        ),
        href="url:N",
        tooltip = [
            alt.Tooltip("comment_:N", title="comment"),
            alt.Tooltip("rating:Q", title="System rating", format=".2f"),
            alt.Tooltip("pred:Q", title="Your rating", format=".2f"),
            alt.Tooltip("category:N", title="Potential toxicity categories")
        ]
    )

    # Filter to specified error type
    if error_type == "System is under-sensitive":
        # FN: system rates non-toxic, but user rates toxic
        chart = chart.transform_filter(
            alt.FieldGTPredicate(field="pred", gt=threshold)
        )
    elif error_type == "System is over-sensitive":
        # FP: system rates toxic, but user rates non-toxic
        chart = chart.transform_filter(
            alt.FieldLTEPredicate(field="pred", lte=threshold)
        )
    
    # Threshold line
    rule = alt.Chart(pd.DataFrame({
        "threshold": [threshold],
    })).mark_rule(color='grey').encode(
        x=alt.X("mean(threshold):Q", scale=alt.Scale(domain=[domain_min, domain_max]), title=""),
        size=alt.value(2),
    )
    
    # Plot region annotations
    nontoxic_x = (domain_min + threshold) / 2.
    toxic_x = (domain_max + threshold) / 2.
    annotation = alt.Chart(pd.DataFrame({
        "annotation_text": ["Non-toxic", "Toxic"],
        "x": [nontoxic_x, toxic_x],
        "y": [max_items, max_items],
    })).mark_text(
        align="center",
        baseline="middle",
        fontSize=16,
        dy=10,
        color="grey"
    ).encode(
        x=alt.X("x", title=""),
        y=alt.Y("y", title="", axis=None),
        text="annotation_text"
    )
    
    # Plot region background colors
    bkgd = alt.Chart(pd.DataFrame({
        "start": [domain_min, threshold],
        "stop": [threshold, domain_max],
        "bkgd": ["Non-toxic (L side)", "Toxic (R side)"],
    })).mark_rect(opacity=1.0, stroke="grey", strokeWidth=0.25).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        x2=alt.X2("stop:Q"),
        y=alt.value(0),
        y2=alt.value(plot_dim_height),
        color=alt.Color("bkgd:O", scale=alt.Scale(
            domain=["Non-toxic (L side)", "Toxic (R side)"], 
            range=["white", "#cbcbcb"]),
            title="Your rating (background color)"
        )
    )
    
    final_plot = (bkgd + annotation + chart + rule).properties(height=(plot_dim_height), width=plot_dim_width).resolve_scale(color='independent').to_json()

    return final_plot, df

def get_cluster_comments(df, error_type, threshold=TOXIC_THRESHOLD, worker_id="A", num_examples=50, use_model=True):    
    df["user_color"] = [get_user_color(user, threshold) for user in df["pred"].tolist()]  # get cell colors
    df["system_color"] = [get_user_color(sys, threshold) for sys in df["rating"].tolist()]  # get cell colors
    df["error_color"] = [get_system_color(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]  # get cell colors
    df["error_type"] = [get_error_type(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]  # get error type in words
    df["error_amt"] = [abs(sys - threshold) for sys in df["rating"].tolist()]  # get raw error
    df["judgment"] = ["" for _ in range(len(df))]  # template for "agree" or "disagree" buttons

    if use_model:
        df = df.sort_values(by=["error_amt"], ascending=False) # surface largest errors first
    else:
        print("get_cluster_comments; not using model")
        df = df.sort_values(by=["rating"], ascending=True)

    df["id"] = df["item_id"]
    # df["comment"] already exists
    df["comment"] = df["comment_"]
    df["toxicity_category"] = df["category"]
    df["user_rating"] = df["pred"]
    df["user_decision"] = [get_decision(rating, threshold) for rating in df["pred"].tolist()]
    df["system_rating"] = df["rating"]
    df["system_decision"] = [get_decision(rating, threshold) for rating in df["rating"].tolist()]
    df["error_type"] = df["error_type"]
    df = df.head(num_examples)
    df = df.round(decimals=2)

    # Filter to specified error type
    if error_type == "System is under-sensitive":
        # FN: system rates non-toxic, but user rates toxic
        df = df[df["error_type"] == "System may be under-sensitive"]
    elif error_type == "System is over-sensitive":
        # FP: system rates toxic, but user rates non-toxic
        df = df[df["error_type"] == "System may be over-sensitive" ]
    elif error_type == "Both":
        df = df[(df["error_type"] == "System may be under-sensitive") | (df["error_type"] == "System may be over-sensitive")]

    return df.to_json(orient="records")

# PERSONALIZED CLUSTERS utils
def get_disagreement_comments(preds_df, mode, n=10_000, threshold=TOXIC_THRESHOLD):
    # Get difference between user rating and system rating
    df = preds_df.copy()
    df["diff"] = [get_error_size(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
    df["error_type"] = [get_error_type(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
    # asc = low to high; lowest = sys lower than user (under-sensitive)
    # desc = high to low; lowest = sys higher than user (over-sensitive)
    if mode == "under-sensitive":
        df = df[df["error_type"] == "System may be under-sensitive"]
        asc = True
    elif mode == "over-sensitive":
        df = df[df["error_type"] == "System may be over-sensitive"]
        asc = False
    df = df.sort_values(by=["diff"], ascending=asc)
    df = df.head(n)
    
    return df["comment_"].tolist(), df

def get_personal_clusters(model, n=3):
    personal_cluster_file = f"./data/personal_cluster_dfs/{model}.pkl"
    if (os.path.isfile(personal_cluster_file)):
        with open(personal_cluster_file, "rb") as f:
            cluster_df = pickle.load(f)
            cluster_df = cluster_df.sort_values(by=["topic_id"])
            topics_under = cluster_df[cluster_df["error_type"] == "System may be under-sensitive"]["topic"].unique().tolist()
            topics_under = topics_under[1:(n + 1)]
            topics_over = cluster_df[cluster_df["error_type"] == "System may be over-sensitive"]["topic"].unique().tolist()
            topics_over = topics_over[1:(n + 1)]
            return topics_under, topics_over
    else:
        topics_under_top = []
        topics_over_top = []
        preds_df_file = f"./data/preds_dfs/{model}.pkl"
        if (os.path.isfile(preds_df_file)):
            with open(preds_df_file, "rb") as f:
                preds_df = pickle.load(f)
                preds_df_mod = preds_df.merge(comments_grouped_full_topic_cat, on="item_id", how="left", suffixes=('_', '_avg')).reset_index()
                preds_df_mod = preds_df_mod[preds_df_mod["user_id"] == "A"]

                comments_under, comments_under_df = get_disagreement_comments(preds_df_mod, mode="under-sensitive", n=1000)
                if len(comments_under) > 0:
                    topics_under = BERTopic(embedding_model="paraphrase-MiniLM-L6-v2").fit(comments_under)
                    topics_under_top = topics_under.get_topic_info().head(n)["Name"].tolist()
                    print("topics_under", topics_under_top)
                    # Get topics per comment
                    topics_assigned, _ = topics_under.transform(comments_under)
                    comments_under_df["topic_id"] = topics_assigned
                    cur_topic_ids = topics_under.get_topic_info().Topic
                    topic_short_names = topics_under.get_topic_info().Name
                    topic_ids_to_names = {cur_topic_ids[i]: topic_short_names[i] for i in range(len(cur_topic_ids))}
                    comments_under_df["topic"] = [topic_ids_to_names[topic_id] for topic_id in comments_under_df["topic_id"].tolist()]

                comments_over, comments_over_df = get_disagreement_comments(preds_df_mod, mode="over-sensitive", n=1000)
                if len(comments_over) > 0:
                    topics_over = BERTopic(embedding_model="paraphrase-MiniLM-L6-v2").fit(comments_over)
                    topics_over_top = topics_over.get_topic_info().head(n)["Name"].tolist()
                    print("topics_over", topics_over_top)
                    # Get topics per comment
                    topics_assigned, _ = topics_over.transform(comments_over)
                    comments_over_df["topic_id"] = topics_assigned
                    cur_topic_ids = topics_over.get_topic_info().Topic
                    topic_short_names = topics_over.get_topic_info().Name
                    topic_ids_to_names = {cur_topic_ids[i]: topic_short_names[i] for i in range(len(cur_topic_ids))}
                    comments_over_df["topic"] = [topic_ids_to_names[topic_id] for topic_id in comments_over_df["topic_id"].tolist()]

                cluster_df = pd.concat([comments_under_df, comments_over_df])
                with open(f"./data/personal_cluster_dfs/{model}.pkl", "wb") as f:
                    pickle.dump(cluster_df, f)

                return topics_under_top, topics_over_top
    return [], []

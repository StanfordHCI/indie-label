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
from datetime import date

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
with open(os.path.join(module_dir, "data/input/ids_to_comments.pkl"), "rb") as f:
    ids_to_comments = pickle.load(f)
with open(os.path.join(module_dir, "data/input/comments_to_ids.pkl"), "rb") as f:
    comments_to_ids = pickle.load(f)
system_preds_df = pd.read_pickle("data/input/system_preds_df.pkl")
sys_eval_df = pd.read_pickle(os.path.join(module_dir, "data/input/split_data/sys_eval_df.pkl"))
train_df = pd.read_pickle(os.path.join(module_dir, "data/input/split_data/train_df.pkl"))
train_df_ids = train_df["item_id"].unique().tolist()
model_eval_df = pd.read_pickle(os.path.join(module_dir, "data/input/split_data/model_eval_df.pkl"))
ratings_df_full = pd.read_pickle(os.path.join(module_dir, "data/input/ratings_df_full.pkl"))
worker_info_df = pd.read_pickle("./data/input/worker_info_df.pkl")

topic_ids = system_preds_df.topic_id
topics = system_preds_df.topic
topic_ids_to_topics = {topic_ids[i]: topics[i] for i in range(len(topic_ids))}
topics_to_topic_ids = {topics[i]: topic_ids[i] for i in range(len(topic_ids))}
unique_topics_ids = sorted(system_preds_df.topic_id.unique())
unique_topics = [topic_ids_to_topics[topic_id] for topic_id in range(len(topic_ids_to_topics) - 1)]

def get_toxic_threshold():
    return TOXIC_THRESHOLD

def get_user_model_names(user):
    # Fetch the user's models
    output_dir = f"./data/output"
    users = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]
    if user not in users:
        # User does not exist
        return []
    else:
        # Fetch trained model names for the user
        user_dir = f"./data/output/{user}"
        user_models = [name for name in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, name))]
        user_models.sort()
        return user_models

def get_unique_topics():
    return unique_topics

def get_large_clusters(min_n):
    counts_df = system_preds_df.groupby(by=["topic_id"]).size().reset_index(name='counts')
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


########################################
# Data storage helper functions
# Set up all directories for new user
def setup_user_dirs(cur_user):
    user_dir = f"./data/output/{cur_user}"
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
def setup_model_dirs(cur_user, cur_model):
    model_dir = f"./data/output/{cur_user}/{cur_model}"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir) # Set up model dir
        # Set up subdirs
        os.mkdir(os.path.join(model_dir, "labels"))
        os.mkdir(os.path.join(model_dir, "perf"))
def setup_user_model_dirs(cur_user, cur_model):
    setup_user_dirs(cur_user)
    setup_model_dirs(cur_user, cur_model)

# Charts
def get_chart_file(cur_user, cur_model):
    chart_dir = f"./data/output/{cur_user}/{cur_model}"
    return os.path.join(chart_dir, f"chart_overall_vis.json")

# Labels
def get_label_dir(cur_user, cur_model):
    return f"./data/output/{cur_user}/{cur_model}/labels"
def get_n_label_files(cur_user, cur_model):
    label_dir = get_label_dir(cur_user, cur_model)
    return len([name for name in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, name))])
def get_label_file(cur_user, cur_model, label_i=None):
    if label_i is None:
        # Get index to add on to end of list
        label_i = get_n_label_files(cur_user, cur_model)
    label_dir = get_label_dir(cur_user, cur_model)
    return os.path.join(label_dir, f"{label_i}.pkl")

# Performance
def get_perf_dir(cur_user, cur_model):
    return f"./data/output/{cur_user}/{cur_model}/perf"
def get_n_perf_files(cur_user, cur_model):
    perf_dir = get_perf_dir(cur_user, cur_model)
    return len([name for name in os.listdir(perf_dir) if os.path.isfile(os.path.join(perf_dir, name))])
def get_perf_file(cur_user, cur_model, perf_i=None):
    if perf_i is None:
        # Get index to add on to end of list
        perf_i = get_n_perf_files(cur_user, cur_model)
    perf_dir = get_perf_dir(cur_user, cur_model)
    return os.path.join(perf_dir, f"{perf_i}.pkl")

# Predictions dataframe
def get_preds_file(cur_user, cur_model):
    preds_dir = f"./data/output/{cur_user}/{cur_model}"
    return os.path.join(preds_dir, f"preds_df.pkl")

# Reports
def get_reports_file(cur_user, cur_model):
    return f"./data/output/{cur_user}/{cur_model}/reports.json"

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

# Generates the summary plot across all topics for the user
def show_overall_perf(cur_model, error_type, cur_user, threshold=TOXIC_THRESHOLD, topic_vis_method="median", use_cache=True):
    # Your perf (calculate using model and testset)    
    preds_file = get_preds_file(cur_user, cur_model)
    with open(preds_file, "rb") as f:
        preds_df = pickle.load(f)

    chart_file = get_chart_file(cur_user, cur_model)
    if use_cache and os.path.isfile(chart_file):
        # Read from file if it exists
        with open(chart_file, "r") as f:
            topic_overview_plot_json = json.load(f)
    else:
        # Otherwise, generate chart and save to file
        if topic_vis_method == "median":  # Default
            preds_df_grp = preds_df.groupby(["topic", "user_id"]).median()
        elif topic_vis_method == "mean":
            preds_df_grp = preds_df.groupby(["topic", "user_id"]).mean()
        topic_overview_plot_json = plot_overall_vis(preds_df=preds_df_grp, n_topics=200, threshold=threshold, error_type=error_type, cur_user=cur_user, cur_model=cur_model)
        # Save to file    
        with open(chart_file, "w") as f:
            json.dump(topic_overview_plot_json, f)

    return {
        "topic_overview_plot_json": json.loads(topic_overview_plot_json),
    }

########################################
# GET_LABELING utils
def create_example_sets(n_label_per_bin, score_bins, keyword=None, topic=None):
    # Restrict to the keyword, if provided
    df = system_preds_df.copy()
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

def get_grp_model_labels(n_label_per_bin, score_bins, grp_ids):
    df = system_preds_df.copy()

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
# SAVE_REPORT utils

# Convert the SEP field selection from the UI to the SEP enum value
def get_sep_enum(sep_selection):
    if sep_selection == "Adversarial Example":
        return "S0403: Adversarial Example"
    elif sep_selection == "Accuracy":
        return "P0204: Accuracy"
    elif sep_selection == "Bias/Discrimination":
        return "E0100: Bias/ Discrimination"
    else:
        return "P0200: Model issues"

# Format the description for the report including the provided title, error type, and text entry field ("Summary/Suggestions" text box)
def format_description(indie_label_json):
    title = indie_label_json["title"]
    error_type = indie_label_json["error_type"]
    text_entry = indie_label_json["text_entry"]
    return f"Title: {title}\nError Type: {error_type}\nSummary/Suggestions: {text_entry}"

# Convert indielabel json to AVID json format.
# See the AVID format in https://avidml.org/avidtools/reference/report
#
# Important mappings:
#   IndieLabel Attribute        AVID Attribute          Example
#   text_entry                  description             "I think the Perspective API
#                                                       is too sensitive. Here are some examples."
#   topic                       feature                 0_shes_woman_lady_face
#   persp_score                 model_score             0.94
#   comment                     ori_input               "She looks beautiful"
#   user_rating                 personal_model_score    0.92
#   user_decision               user_decision           "Non-toxic"
# Note that this is at the individual report level.
def convert_indie_label_json_to_avid_json(indie_label_json, cur_user, email, sep_selection):

    # Setting up the structure with a dict to enable programmatic additions
    avid_json_dict = { 
        "data_type": "AVID",
        "data_version": None,
        "metadata": None,
        "affects": {
            "developer": [],
            "deployer": [
              "Hugging Face"
            ],
            # TODO: Make artifacts malleable during modularity work
            "artifacts": [
              {
                "type": "Model",
                "name": "Perspective API"
              }
            ]
        },
        "problemtype": {
            "classof": "Undefined", # I don't think any of the other ClassEnums are applicable. Link: https://avidml.org/avidtools/_modules/avidtools/datamodels/enums#ClassEnum
            "type": "Detection",
            "description": {
              "lang": "eng", # TODO: Make language selectable
              "value": "This report contains results from an end user audit conducted on Hugging Face."
            }
          },
        "metrics": [ # Note: For the end users use case, I made each comment an example.
          ],
        "references": [],
        "description": {
            "lang": "eng", # TODO: Make language selectable
            "value": "" # Leaving empty so the report comments can be contained here.
          },
          "impact": {
            "avid": {
              "risk_domain": [
                "Ethics"
              ],
              "sep_view": [
                "E0101: Group fairness"
              ],
              "lifecycle_view": [
                "L05: Evaluation"
              ],
              "taxonomy_version": "0.2"
            }
          },
          "credit": "", # Leaving empty so that credit can be assigned
          "reported_date": "" # Leaving empty so that it can be dynamically filled in
    }

    avid_json_dict["description"] = format_description(indie_label_json)
    avid_json_dict["reported_date"] = str(date.today())
    # Assign credit to email if provided, otherwise default to randomly assigned username
    if email != "":
        avid_json_dict["credit"] = email
    else:
        avid_json_dict["credit"] = cur_user

    sep_enum = get_sep_enum(sep_selection)
    avid_json_dict["impact"]["avid"]["sep_view"] = [sep_enum]

    for e in indie_label_json["evidence"]:
        curr_metric = {}
        curr_metric["name"] = "Perspective API"
        curr_metric["detection_method"] = {
            "type": "Detection",
            "name": "Individual Example from End User Audit"
        }
        res_dict = {}
        res_dict["feature"] = e["topic"]
        res_dict["model_score"] = str(e["persp_score"]) # Converted to string to avoid Float type error with DB
        res_dict["ori_input"] = e["comment"]
        res_dict["personal_model_score"] = str(e["user_rating"]) # See above
        res_dict["user_decision"] = e["user_decision"]
        curr_metric["results"] = res_dict
        avid_json_dict["metrics"].append(curr_metric)

    new_report = json.dumps(avid_json_dict)
    return new_report

########################################
# GET_PERSONALIZED_MODEL utils
def fetch_existing_data(user, model_name):
    # Check if we have cached model performance
    n_perf_files = get_n_perf_files(user, model_name)
    if n_perf_files > 0:
        # Fetch cached results
        perf_file = get_perf_file(user, model_name, n_perf_files - 1)  # Get last performance file
        with open(perf_file, "rb") as f:
            mae, mse, rmse, avg_diff = pickle.load(f)
    else:
        raise Exception(f"Model {model_name} does not exist")
    
    # Fetch previous user-provided labels
    ratings_prev = None
    n_label_files = get_n_label_files(user, model_name)
    if n_label_files > 0:
        label_file = get_label_file(user, model_name, n_label_files - 1) # Get last label file
        with open(label_file, "rb") as f:
            ratings_prev = pickle.load(f)
    return mae, mse, rmse, avg_diff, ratings_prev

# Main function called by server's `get_personalized_model` endpoint
# Trains an updated model with the specified name, user, and ratings
# Saves ratings, performance metrics, and pre-computed predictions to files
# - model_name: name of the model to train
# - ratings: dictionary of comments to ratings
# - user: user name
# - top_n: number of comments to train on (used when a set was held out for original user study)
# - topic: topic to train on (used when tuning for a specific topic)
def train_updated_model(model_name, ratings, user, top_n=None, topic=None, debug=False):
    # Check if there is previously-labeled data; if so, combine it with this data
    labeled_df = format_labeled_data(ratings, worker_id=user) # Treat ratings as full batch of all ratings
    ratings_prev = None

    # Filter out rows with "unsure" (-1)
    labeled_df = labeled_df[labeled_df["rating"] != -1]

    # Filter to top N for user study
    if (topic is None) and (top_n is not None):
        labeled_df = labeled_df.head(top_n)
    else:
        # For topic tuning, need to fetch old labels
        n_label_files = get_n_label_files(user, model_name)
        if n_label_files > 0:
            # Concatenate previous set of labels with this new batch of labels
            label_file = get_label_file(user, model_name, n_label_files - 1) # Get last label file
            with open(label_file, "rb") as f:
                ratings_prev = pickle.load(f)
                labeled_df_prev = format_labeled_data(ratings_prev, worker_id=user)
                labeled_df_prev = labeled_df_prev[labeled_df_prev["rating"] != -1]
                ratings.update(ratings_prev) # append old ratings to ratings
                labeled_df = pd.concat([labeled_df_prev, labeled_df])
    if debug:
        print("len ratings for training:", len(labeled_df))
    # Save this batch of labels
    label_file = get_label_file(user, model_name)
    with open(label_file, "wb") as f:
        pickle.dump(ratings, f)

    # Train model
    cur_model, _, _, _ = train_user_model(ratings_df=labeled_df)
    
    # Compute performance metrics
    mae, mse, rmse, avg_diff = users_perf(cur_model, worker_id=user)
    # Save performance metrics
    perf_file = get_perf_file(user, model_name)
    with open(perf_file, "wb") as f:
        pickle.dump((mae, mse, rmse, avg_diff), f)

    # Pre-compute predictions for full dataset
    cur_preds_df = get_preds_df(cur_model, [user], sys_eval_df=ratings_df_full)
    # Save pre-computed predictions
    preds_file = get_preds_file(user, model_name)
    with open(preds_file, "wb") as f:
        pickle.dump(cur_preds_df, f)

    # Replace cached summary plot if it exists
    show_overall_perf(cur_model=model_name, error_type="Both", cur_user=user, use_cache=False)

    ratings_prev = ratings
    return mae, mse, rmse, avg_diff, ratings_prev

def format_labeled_data(ratings, worker_id):    
    all_rows = []
    for comment, rating in ratings.items():
        comment_id = comments_to_ids[comment]
        row = [worker_id, comment_id, int(rating)]
        all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=["user_id", "item_id", "rating"])
    return df

def users_perf(model, worker_id, sys_eval_df=sys_eval_df):
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

    df = df.merge(system_preds_df, on="item_id", how="left", suffixes=('', '_sys'))
    df.dropna(subset = ["pred"], inplace=True)
    df["rating"] = df.rating.astype("int32")

    perf_metrics = get_overall_perf(df, worker_id) # mae, mse, rmse, avg_diff  
    return perf_metrics

def get_overall_perf(preds_df, user_id):    
    # Prepare dataset to calculate performance
    y_pred = preds_df[preds_df["user_id"] == user_id].rating_sys.to_numpy() # system's prediction
    y_true = preds_df[preds_df["user_id"] == user_id].pred.to_numpy() # user's (predicted) ground truth
    
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
# - sys_eval_df: dataframe of system eval labels (pre-computed)
def get_preds_df(model, user_ids, sys_eval_df=sys_eval_df, bins=BINS, debug=False):
    # Prep dataframe for all predictions we'd like to request
    start = time.time()
    sys_eval_comment_ids = sys_eval_df.item_id.unique().tolist()

    empty_ratings_rows = []
    for user_id in user_ids:
        empty_ratings_rows.extend([[user_id, c_id, 0] for c_id in sys_eval_comment_ids])
    empty_ratings_df = pd.DataFrame(empty_ratings_rows, columns=["user_id", "item_id", "rating"])
    if debug:
        print("setup", time.time() - start)
    
    # Evaluate model to get predictions
    start = time.time() 
    reader = Reader(rating_scale=(0, 4))
    eval_set_data = Dataset.load_from_df(empty_ratings_df, reader)
    _, testset = train_test_split(eval_set_data, test_size=1.)
    predictions = model.test(testset)
    if debug:
        print("train_test_split", time.time() - start)
    
    # Update dataframe with predictions
    start = time.time()
    df = empty_ratings_df.copy() # user_id, item_id, rating
    user_item_preds = get_predictions_by_user_and_item(predictions)
    df["pred"] = df.apply(lambda row: user_item_preds[(row.user_id, row.item_id)] if (row.user_id, row.item_id) in user_item_preds else np.nan, axis=1)
    df = df.merge(system_preds_df, on="item_id", how="left", suffixes=('', '_sys'))
    df.dropna(subset = ["pred"], inplace=True)
    df["rating"] = df.rating.astype("int32")
    
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
def train_model(train_df, model_eval_df, model_type="SVD", sim_type=None, user_based=True, debug=False):
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
    
    if debug:
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, FCP: {fcp}")
    perf = [mae, mse, rmse, fcp]
    
    return algo, perf

def plot_train_perf_results(user, model_name, mae):
    n_perf_files = get_n_perf_files(user, model_name)
    all_rows = []
    for i in range(n_perf_files):
        perf_file = get_perf_file(user, model_name, i)
        with open(perf_file, "rb") as f:
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

    # Manually set for now
    mae_good = 1.0
    mae_okay = 1.2

    plot_dim_width = 500
    domain_min = 0.0
    domain_max = 2.0
    bkgd = alt.Chart(pd.DataFrame({
        "start": [mae_okay, mae_good, domain_min],
        "stop": [domain_max, mae_okay, mae_good],
        "bkgd": ["Needs improvement", "Okay", "Good"],
    })).mark_rect(opacity=0.2).encode(
        y=alt.Y("start:Q", scale=alt.Scale(domain=[0, domain_max]), title=""),
        y2=alt.Y2("stop:Q", title="Performance (MAE)"),
        x=alt.value(0),
        x2=alt.value(plot_dim_width),
        color=alt.Color("bkgd:O", scale=alt.Scale(
            domain=["Needs improvement", "Okay", "Good"], 
            range=["red", "yellow", "green"]),
            title="How good is your MAE?"
        )
    )

    plot = (bkgd + chart).properties(width=plot_dim_width).resolve_scale(color='independent')
    mae_status = None
    if mae < mae_good:
        mae_status = "Your MAE is in the <b>Good</b> range. Your model looks ready to go."
    elif mae < mae_okay:
        mae_status = "Your MAE is in the <b>Okay</b> range. Your model can be used, but you can provide additional labels to improve it."
    else:
        mae_status = "Your MAE is in the <b>Needs improvement</b> range. Your model may need additional labels to improve."
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
        "is_profane_frac": "Profanity", 
        "is_threat_frac": "Threat", 
        "is_identity_attack_frac": "Identity Attack", 
        "is_insult_frac": "Insult", 
        "is_sexual_harassment_frac": "Sexual Harassment",
    }
    categories = []
    for k in ["is_profane_frac", "is_threat_frac", "is_identity_attack_frac", "is_insult_frac", "is_sexual_harassment_frac"]:
        if row[k] > threshold:
            categories.append(k_to_category[k])
    
    if len(categories) > 0:
        return ", ".join(categories)
    else:
        return ""

def get_comment_url(row):
    return f"#{row['item_id']}/#comment"

def get_topic_url(row):
    return f"#{row['topic']}/#topic"

# Plots overall results histogram (each block is a topic)
def plot_overall_vis(preds_df, error_type, cur_user, cur_model, n_topics=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, sys_col="rating_sys"):
    df = preds_df.copy().reset_index()
    
    if n_topics is not None:
        df = df[df["topic_id"] < n_topics]
    
    df["vis_pred_bin"], out_bins = pd.cut(df["pred"], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == cur_user].sort_values(by=["item_id"]).reset_index()
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df[sys_col].tolist()]
    df["threshold"] = [threshold for r in df[sys_col].tolist()]
    df["key"] = [get_key(sys, user, threshold) for sys, user in zip(df[sys_col].tolist(), df["pred"].tolist())]
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
        sort=[{'field': sys_col}],
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
            alt.Tooltip("topic:N", title="Topic"),
            alt.Tooltip("system_label:N", title="System label"),
            alt.Tooltip(f"{sys_col}:Q", title="System rating", format=".2f"),
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
    return plot

# Plots cluster results histogram (each block is a comment), but *without* a model 
# as a point of reference (in contrast to plot_overall_vis_cluster)
def plot_overall_vis_cluster_no_model(cur_user, preds_df, n_comments=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, sys_col="rating_sys"):
    df = preds_df.copy().reset_index()
    
    df["vis_pred_bin"], out_bins = pd.cut(df[sys_col], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == cur_user].sort_values(by=[sys_col]).reset_index()
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df[sys_col].tolist()]
    df["key"] = [get_key_no_model(sys, threshold) for sys in df[sys_col].tolist()]
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
        sort=[{'field': sys_col}],
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
            alt.Tooltip("comment:N", title="comment"),
            alt.Tooltip(f"{sys_col}:Q", title="System rating", format=".2f"),
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
def plot_overall_vis_cluster(cur_user, preds_df, error_type, n_comments=None, bins=VIS_BINS, threshold=TOXIC_THRESHOLD, sys_col="rating_sys"):
    df = preds_df.copy().reset_index()
    
    df["vis_pred_bin"], out_bins = pd.cut(df["pred"], bins, labels=VIS_BINS_LABELS, retbins=True)
    df = df[df["user_id"] == cur_user].sort_values(by=[sys_col]).reset_index(drop=True)
    df["system_label"] = [("toxic" if r > threshold else "non-toxic") for r in df[sys_col].tolist()]
    df["key"] = [get_key(sys, user, threshold) for sys, user in zip(df[sys_col].tolist(), df["pred"].tolist())]
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
        sort=[{'field': sys_col}],
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
            alt.Tooltip("comment:N", title="comment"),
            alt.Tooltip(f"{sys_col}:Q", title="System rating", format=".2f"),
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

def get_cluster_comments(df, error_type, threshold=TOXIC_THRESHOLD, sys_col="rating_sys", use_model=True, debug=False):    
    df["user_color"] = [get_user_color(user, threshold) for user in df["pred"].tolist()]  # get cell colors
    df["system_color"] = [get_user_color(sys, threshold) for sys in df[sys_col].tolist()]  # get cell colors
    df["error_color"] = [get_system_color(sys, user, threshold) for sys, user in zip(df[sys_col].tolist(), df["pred"].tolist())]  # get cell colors
    df["error_type"] = [get_error_type(sys, user, threshold) for sys, user in zip(df[sys_col].tolist(), df["pred"].tolist())]  # get error type in words
    df["error_amt"] = [abs(sys - threshold) for sys in df[sys_col].tolist()]  # get raw error
    df["judgment"] = ["" for _ in range(len(df))]  # template for "agree" or "disagree" buttons

    if use_model:
        df = df.sort_values(by=["error_amt"], ascending=False) # surface largest errors first
    else:
        if debug:
            print("get_cluster_comments; not using model")
        df = df.sort_values(by=[sys_col], ascending=True)

    df["id"] = df["item_id"]
    df["toxicity_category"] = df["category"]
    df["user_rating"] = df["pred"]
    df["user_decision"] = [get_decision(rating, threshold) for rating in df["pred"].tolist()]
    df["system_rating"] = df[sys_col]
    df["system_decision"] = [get_decision(rating, threshold) for rating in df[sys_col].tolist()]
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

    return df

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
    
    return df["comment"].tolist(), df

def get_explore_df(n_examples, threshold):
    df = system_preds_df.sample(n=n_examples)
    df["system_decision"] = [get_decision(rating, threshold) for rating in df["rating"].tolist()]
    df["system_color"] = [get_user_color(sys, threshold) for sys in df["rating"].tolist()]  # get cell colors
    return df
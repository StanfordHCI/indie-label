from flask import Flask, send_from_directory
from flask import request

import random
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import math
import altair as alt
import matplotlib.pyplot as plt
import time
import friendlywords as fw

import audit_utils as utils

import requests


app = Flask(__name__)
DEBUG = False  # Debug flag for development; set to False for production

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('indie_label_svelte/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('indie_label_svelte/public', path)


########################################
# ROUTE: /AUDIT_SETTINGS

@app.route("/audit_settings")
def audit_settings(debug=DEBUG):
    # Fetch page content
    user = request.args.get("user")
    scaffold_method = request.args.get("scaffold_method")

    # Assign user ID if none is provided (default case)
    if user == "null":
        # Generate random two-word user ID
        user = fw.generate(2, separator="_")

    user_models = utils.get_user_model_names(user)
    grp_models = [m for m in user_models if m.startswith(f"model_{user}_group_")]

    clusters = utils.get_unique_topics()
    if len(user_models) > 2 and scaffold_method != "tutorial" and user != "DemoUser":
        # Highlight topics that have been tuned
        tuned_clusters = [m.lstrip(f"model_{user}_") for m in user_models if (m != f"model_{user}" and not m.startswith(f"model_{user}_group_"))]
        other_clusters = [c for c in clusters if c not in tuned_clusters]
        tuned_options = {
            "label": "Topics with tuned models",
            "options": [{"value": i, "text": cluster} for i, cluster in enumerate(tuned_clusters)],
        }
        other_options = {
            "label": "All other topics",
            "options": [{"value": i, "text": cluster} for i, cluster in enumerate(other_clusters)],
        }
        clusters_options = [tuned_options, other_options]
    else:
        clusters_options = [{
            "label": "All auto-generated topics",
            "options": [{"value": i, "text": cluster} for i, cluster in enumerate(clusters)],
        },]

    clusters_for_tuning = utils.get_large_clusters(min_n=150)
    clusters_for_tuning_options = [{"value": i, "text": cluster} for i, cluster in enumerate(clusters_for_tuning)]  # Format for Svelecte UI element

    context = {
        "personalized_models": user_models,
        "personalized_model_grp": grp_models,
        "perf_metrics": ["Average rating difference", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)"],
        "clusters": clusters_options,
        "clusters_for_tuning": clusters_for_tuning_options,
        "user": user,
    }
    return json.dumps(context)


########################################
# ROUTE: /GET_AUDIT
@app.route("/get_audit")
def get_audit():
    pers_model = request.args.get("pers_model")
    error_type = request.args.get("error_type")
    cur_user = request.args.get("cur_user")
    topic_vis_method = request.args.get("topic_vis_method") 
    if topic_vis_method == "null":
        topic_vis_method = "median"

    if pers_model == "" or pers_model == "null" or pers_model == "undefined":
        overall_perf = None
    else:
        overall_perf = utils.show_overall_perf(
            cur_model=pers_model,
            error_type=error_type,
            cur_user=cur_user,
            topic_vis_method=topic_vis_method,
        )

    results = {
        "overall_perf": overall_perf,
    }
    return json.dumps(results)

########################################
# ROUTE: /GET_CLUSTER_RESULTS
@app.route("/get_cluster_results")
def get_cluster_results(debug=DEBUG):
    pers_model = request.args.get("pers_model")
    cur_user = request.args.get("cur_user")
    cluster = request.args.get("cluster")
    topic_df_ids = request.args.getlist("topic_df_ids")
    topic_df_ids = [int(val) for val in topic_df_ids[0].split(",") if val != ""]
    search_type = request.args.get("search_type")
    keyword = request.args.get("keyword")
    error_type = request.args.get("error_type")
    use_model = request.args.get("use_model") == "true"

    if debug:
        print(f"get_cluster_results using model {pers_model}")

    # Prepare cluster df (topic_df)
    topic_df = None
    preds_file = utils.get_preds_file(cur_user, pers_model)
    with open(preds_file, "rb") as f:
        topic_df = pickle.load(f)
    if search_type == "cluster":
        # Display examples with comment, your pred, and other users' pred
        topic_df = topic_df[(topic_df["topic"] == cluster) | (topic_df["item_id"].isin(topic_df_ids))]
    elif search_type == "keyword":
        topic_df = topic_df[(topic_df["comment"].str.contains(keyword, case=False, regex=False)) | (topic_df["item_id"].isin(topic_df_ids))]

    topic_df = topic_df.drop_duplicates()
    if debug:
        print("len topic_df", len(topic_df))

    # Handle empty results
    if len(topic_df) == 0: 
        results = {
            "user_perf_rounded": None,
            "user_direction": None,
            "other_perf_rounded": None,
            "other_direction": None,
            "n_other_users": None,
            "cluster_examples": None,
            "odds_ratio": None,
            "odds_ratio_explanation": None,
            "topic_df_ids": [],
            "cluster_overview_plot_json": None,
            "cluster_comments": None, 
        }
        return results

    topic_df_ids = topic_df["item_id"].unique().tolist()

    # Prepare overview plot for the cluster
    if use_model:
        # Display results with the model as a reference point
        cluster_overview_plot_json, sampled_df = utils.plot_overall_vis_cluster(cur_user, topic_df, error_type=error_type, n_comments=500)
    else:
        # Display results without a model
        cluster_overview_plot_json, sampled_df = utils.plot_overall_vis_cluster_no_model(cur_user, topic_df, n_comments=500)

    cluster_comments = utils.get_cluster_comments(sampled_df,error_type=error_type, use_model=use_model)  # New version of cluster comment table

    results = {
        "topic_df_ids": topic_df_ids,
        "cluster_overview_plot_json": json.loads(cluster_overview_plot_json),
        "cluster_comments": cluster_comments.to_json(orient="records"), 
    }
    return json.dumps(results)

########################################
# ROUTE: /GET_GROUP_SIZE
@app.route("/get_group_size")
def get_group_size():
    # Fetch info for initial labeling component
    sel_gender = request.args.get("sel_gender")
    sel_pol = request.args.get("sel_pol")
    sel_relig = request.args.get("sel_relig")
    sel_race = request.args.get("sel_race")
    sel_lgbtq = request.args.get("sel_lgbtq")
    if sel_race != "":
        sel_race = sel_race.split(",")

    _, group_size = utils.get_workers_in_group(sel_gender, sel_race, sel_relig, sel_pol, sel_lgbtq)

    context = {
        "group_size": group_size,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_GROUP_MODEL
@app.route("/get_group_model")
def get_group_model(debug=DEBUG):
    # Fetch info for initial labeling component
    model_name = request.args.get("model_name")
    user = request.args.get("user")
    sel_gender = request.args.get("sel_gender")
    sel_pol = request.args.get("sel_pol")
    sel_relig = request.args.get("sel_relig")
    sel_lgbtq = request.args.get("sel_lgbtq")
    sel_race_orig = request.args.get("sel_race")
    if sel_race_orig != "":
        sel_race = sel_race_orig.split(",")
    else:
        sel_race = ""
    start = time.time()

    grp_df, group_size = utils.get_workers_in_group(sel_gender, sel_race, sel_relig, sel_pol, sel_lgbtq)

    grp_ids = grp_df["worker_id"].tolist()

    ratings_grp = utils.get_grp_model_labels(
        n_label_per_bin=BIN_DISTRIB,
        score_bins=SCORE_BINS,
        grp_ids=grp_ids,
    )

    # Modify model name
    model_name = f"{model_name}_group_gender{sel_gender}_relig{sel_relig}_pol{sel_pol}_race{sel_race_orig}_lgbtq_{sel_lgbtq}"
    utils.setup_user_model_dirs(user, model_name)

    # Train group model
    mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, ratings_grp, user)

    duration = time.time() - start
    if debug:
        print("Time to train/cache:", duration)

    context = {
        "group_size": group_size,
        "mae": mae,
    }
    return json.dumps(context)
    
########################################
# ROUTE: /GET_LABELING
@app.route("/get_labeling")
def get_labeling():
    # Fetch info for initial labeling component
    user = request.args.get("user")

    clusters_for_tuning = utils.get_large_clusters(min_n=150)
    clusters_for_tuning_options = [{"value": i, "text": cluster} for i, cluster in enumerate(clusters_for_tuning)]  # Format for Svelecte UI element

    model_name_suggestion = f"my_model"

    context = {
        "personalized_models": utils.get_user_model_names(user),
        "model_name_suggestion": model_name_suggestion,
        "clusters_for_tuning": clusters_for_tuning_options,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_COMMENTS_TO_LABEL
if DEBUG:
    BIN_DISTRIB = [1, 2, 4, 2, 1]  # 10 comments
else:
    BIN_DISTRIB = [2, 4, 8, 4, 2]  # 20 comments
SCORE_BINS = [(0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.01)]
@app.route("/get_comments_to_label")
def get_comments_to_label():
    n = int(request.args.get("n"))
    # Fetch examples to label
    to_label_ids = utils.create_example_sets(
        n_label_per_bin=BIN_DISTRIB,
        score_bins=SCORE_BINS,
        keyword=None
    )
    random.shuffle(to_label_ids)  # randomize to not prime users
    to_label_ids = to_label_ids[:n]

    ids_to_comments = utils.get_ids_to_comments()
    to_label = [ids_to_comments[comment_id] for comment_id in to_label_ids]
    context = {
        "to_label": to_label,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_COMMENTS_TO_LABEL_TOPIC
@app.route("/get_comments_to_label_topic")
def get_comments_to_label_topic():
    # Fetch examples to label
    topic = request.args.get("topic")
    to_label_ids = utils.create_example_sets(
        n_label_per_bin=BIN_DISTRIB,
        score_bins=SCORE_BINS,
        keyword=None,
        topic=topic,
    )
    random.shuffle(to_label_ids)  # randomize to not prime users 
    ids_to_comments = utils.get_ids_to_comments()
    to_label = [ids_to_comments[comment_id] for comment_id in to_label_ids]
    context = {
        "to_label": to_label,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_PERSONALIZED_MODEL
@app.route("/get_personalized_model")
def get_personalized_model(debug=DEBUG):
    model_name = request.args.get("model_name")
    ratings_json = request.args.get("ratings")
    mode = request.args.get("mode")
    user = request.args.get("user")
    ratings = json.loads(ratings_json)
    if debug:
        print(ratings)
        start = time.time()

    utils.setup_user_model_dirs(user, model_name)

    # Handle existing or new model cases
    if mode == "view":
        # Fetch prior model performance
        mae, mse, rmse, avg_diff, ratings_prev = utils.fetch_existing_data(user, model_name)
        
    elif mode == "train":
        # Train model and cache predictions using new labels
        print("get_personalized_model train")
        mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, ratings, user)
    
    if debug:
        duration = time.time() - start
        print("Time to train/cache:", duration) 

    perf_plot, mae_status = utils.plot_train_perf_results(user, model_name, mae)
    perf_plot_json = perf_plot.to_json()

    def round_metric(x):
        return np.round(abs(x), 3)

    results = {
        "model_name": model_name,
        "mae": round_metric(mae),
        "mae_status": mae_status,
        "mse": round_metric(mse),
        "rmse": round_metric(rmse),
        "avg_diff": round_metric(avg_diff),
        "ratings_prev": ratings_prev,
        "perf_plot_json": json.loads(perf_plot_json),
    }
    return json.dumps(results)


########################################
# ROUTE: /GET_PERSONALIZED_MODEL_TOPIC
@app.route("/get_personalized_model_topic")
def get_personalized_model_topic(debug=DEBUG):
    model_name = request.args.get("model_name")
    ratings_json = request.args.get("ratings")
    user = request.args.get("user")
    ratings = json.loads(ratings_json)
    topic = request.args.get("topic")
    if debug:
        print(ratings)
    start = time.time()

    # Modify model name
    model_name = f"{model_name}_{topic}"
    utils.setup_user_model_dirs(user, model_name)

    # Handle existing or new model cases
    # Train model and cache predictions using new labels
    if debug:
        print("get_personalized_model_topic train")
    mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, ratings, user, topic=topic)
        
    if debug:
        duration = time.time() - start
        print("Time to train/cache:", duration) 

    results = {
        "success": "success",
        "ratings_prev": ratings_prev,
        "new_model_name": model_name,
    }
    return json.dumps(results)


########################################
# ROUTE: /GET_REPORTS
@app.route("/get_reports")
def get_reports():
    cur_user = request.args.get("cur_user")
    scaffold_method = request.args.get("scaffold_method")
    model = request.args.get("model")
    topic_vis_method = request.args.get("topic_vis_method")
    if topic_vis_method == "null":
        topic_vis_method = "fp_fn"

    # Load reports for current user from stored file
    reports_file = utils.get_reports_file(cur_user, model)
    if not os.path.isfile(reports_file):
        if scaffold_method == "fixed":
            reports = get_fixed_scaffold()
        elif (scaffold_method == "personal" or scaffold_method == "personal_group" or scaffold_method == "personal_test"):
            reports = get_personal_scaffold(cur_user, model, topic_vis_method)
        elif scaffold_method == "prompts":
            reports = get_prompts_scaffold()
        elif scaffold_method == "tutorial":
            reports = get_tutorial_scaffold()
        else:
            # Prepare empty report
            reports = [
                {
                    "title": "",
                    "error_type": "",
                    "evidence": [],
                    "text_entry": "",
                    "sep_selection": "",
                    "complete_status": False,
                }
            ]
    else:
        # Load from pickle file
        with open(reports_file, "rb") as f:
            reports = json.load(f)

    results = {
        "reports": reports,
    }
    return json.dumps(results)

def get_fixed_scaffold():
    return [
        {
            "title": "Topic: 6_jews_jew_jewish_rabbi",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 73_troll_trolls_trolling_spammers",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 66_mexicans_mexico_mexican_spanish",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 89_cowards_coward_cowardly_brave",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 63_disgusting_gross_toxic_thicc",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
    ]

def get_empty_report(title, error_type):
    return {
            "title": f"Topic: {title}",
            "error_type": error_type,
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        }

def get_tutorial_scaffold():
    return [
        {
            "title": "Topic: 79_idiot_dumb_stupid_dumber",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
    ] 

def get_topic_errors(df, topic_vis_method, threshold=2):
    topics = df["topic"].unique().tolist()
    topic_errors = {}
    for topic in topics:
        t_df = df[df["topic"] == topic]
        y_true = t_df["pred"].to_numpy()  # Predicted user rating (treated as ground truth)
        y_pred = t_df["rating_sys"].to_numpy()  # System rating (which we're auditing)
        if topic_vis_method == "mae":
            t_err = mean_absolute_error(y_true, y_pred)
        elif topic_vis_method == "mse":
            t_err = mean_squared_error(y_true, y_pred)
        elif topic_vis_method == "avg_diff":
            t_err = np.mean(y_true - y_pred)
        elif topic_vis_method == "fp_proportion":
            y_true = [0 if rating < threshold else 1 for rating in y_true]
            y_pred = [0 if rating < threshold else 1 for rating in y_pred]
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except:
                tn, fp, fn, tp = [0, 0, 0, 0]  # ignore; set error to 0
            total = float(len(y_true))
            t_err = fp / total
        elif topic_vis_method == "fn_proportion":
            y_true = [0 if rating < threshold else 1 for rating in y_true]
            y_pred = [0 if rating < threshold else 1 for rating in y_pred]
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except:
                tn, fp, fn, tp = [0, 0, 0, 0]  # ignore; set error to 0
            total = float(len(y_true))
            t_err = fn / total
        topic_errors[topic] = t_err
        
    return topic_errors

def get_personal_scaffold(cur_user, model, topic_vis_method, n_topics=200, n=5, debug=DEBUG):
    threshold = utils.get_toxic_threshold()

    # Get topics with greatest amount of error
    preds_file = utils.get_preds_file(cur_user, model)
    with open(preds_file, "rb") as f:
        preds_df = pickle.load(f)
        preds_df_mod = preds_df[preds_df["user_id"] == cur_user].sort_values(by=["item_id"]).reset_index()
        preds_df_mod = preds_df_mod[preds_df_mod["topic_id"] < n_topics]

        if topic_vis_method == "median":
            df = preds_df_mod.groupby(["topic", "user_id"]).median().reset_index()
        elif topic_vis_method == "mean":
            df = preds_df_mod.groupby(["topic", "user_id"]).mean().reset_index()
        elif topic_vis_method == "fp_fn":
            for error_type in ["fn_proportion", "fp_proportion"]:
                topic_errors = get_topic_errors(preds_df_mod, error_type)
                preds_df_mod[error_type] = [topic_errors[topic] for topic in preds_df_mod["topic"].tolist()]
            df = preds_df_mod.groupby(["topic", "user_id"]).mean().reset_index()
        else:
            # Get error for each topic
            topic_errors = get_topic_errors(preds_df_mod, topic_vis_method)
            preds_df_mod[topic_vis_method] = [topic_errors[topic] for topic in preds_df_mod["topic"].tolist()]
            df = preds_df_mod.groupby(["topic", "user_id"]).mean().reset_index()

        # Get system error
        junk_topics = ["53_maiareficco_kallystas_dyisisitmanila_tractorsazi", "-1_dude_bullshit_fight_ain"]
        df = df[~df["topic"].isin(junk_topics)]  # Exclude known "junk topics"
        
        if topic_vis_method == "median" or topic_vis_method == "mean":
            df["error_magnitude"] = [utils.get_error_magnitude(sys, user, threshold) for sys, user in zip(df["rating_sys"].tolist(), df["pred"].tolist())]
            df["error_type"] = [utils.get_error_type_radio(sys, user, threshold) for sys, user in zip(df["rating_sys"].tolist(), df["pred"].tolist())]

            df_under = df[df["error_type"] == "System is under-sensitive"]
            df_under = df_under.sort_values(by=["error_magnitude"], ascending=False).head(n) # surface largest errors first
            report_under = [get_empty_report(row["topic"], row["error_type"]) for _, row in df_under.iterrows()]

            df_over = df[df["error_type"] == "System is over-sensitive"]
            df_over = df_over.sort_values(by=["error_magnitude"], ascending=False).head(n) # surface largest errors first
            report_over = [get_empty_report(row["topic"], row["error_type"]) for _, row in df_over.iterrows()]
            
            # Set up reports
            reports = (report_under + report_over)
            random.shuffle(reports)
        elif topic_vis_method == "fp_fn":
            df_under = df.sort_values(by=["fn_proportion"], ascending=False).head(n)
            df_under = df_under[df_under["fn_proportion"] > 0]
            if debug:
                print(df_under[["topic", "fn_proportion"]])
            report_under = [get_empty_report(row["topic"], "System is under-sensitive") for _, row in df_under.iterrows()]
            
            df_over = df.sort_values(by=["fp_proportion"], ascending=False).head(n)
            df_over = df_over[df_over["fp_proportion"] > 0]
            if debug:
                print(df_over[["topic", "fp_proportion"]])
            report_over = [get_empty_report(row["topic"], "System is over-sensitive") for _, row in df_over.iterrows()]

            reports = (report_under + report_over)
            random.shuffle(reports)
        else:
            df = df.sort_values(by=[topic_vis_method], ascending=False).head(n * 2)
            df["error_type"] = [utils.get_error_type_radio(sys, user, threshold) for sys, user in zip(df["rating_sys"].tolist(), df["pred"].tolist())]
            reports = [get_empty_report(row["topic"], row["error_type"]) for _, row in df.iterrows()]

        return reports

def get_prompts_scaffold():
    return [
        {
            "title": "Are there terms that are used in your identity group or community that tend to be flagged incorrectly as toxic?",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Are there terms that are used in your identity group or community that tend to be flagged incorrectly as non-toxic?",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Are there certain ways that your community tends to be targeted by outsiders?",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Are there other communities whose content should be very similar to your community's? Verify that this content is treated similarly by the system.",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
        {
            "title": "Are there ways that you've seen individuals in your community actively try to thwart the rules of automated content moderation systems? Check whether these strategies work here.",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "sep_selection": "",
            "complete_status": False,
        },
    ]

# Filter to eligible reports: those that have been marked complete and include at least one piece of evidence.
def get_eligible_reports(reports):
    eligible_reports = []
    for r in reports:
        if (r["complete_status"] == True) and (len(r["evidence"]) > 0):
            eligible_reports.append(r)
    return eligible_reports

# Submit all reports to AVID
# Logs the responses
def submit_reports_to_AVID(reports, cur_user, name, email, debug=DEBUG):
    # Set up the connection to AVID
    root = os.environ.get('AVID_API_URL')
    api_key = os.environ.get('AVID_API_KEY')
    key = {"Authorization": api_key}

    reports = get_eligible_reports(reports)
    if debug:
        print("Num eligible reports:", len(reports))
    
    for r in reports:
        sep_selection = r["sep_selection"]
        new_report = utils.convert_indie_label_json_to_avid_json(r, cur_user, name, email, sep_selection)
        url = root + "submit"
        response = requests.post(url, json=json.loads(new_report), headers=key) # The loads ensures type compliance
        uuid = response.json()
        if debug:
            print("Report", new_report)
            print("AVID API response:", response, uuid)

########################################
# ROUTE: /SAVE_REPORTS
@app.route("/save_reports")
def save_reports(debug=DEBUG):
    cur_user = request.args.get("cur_user")
    reports_json = request.args.get("reports")
    reports = json.loads(reports_json)
    model = request.args.get("model")

    # Save reports for current user to file
    reports_file = utils.get_reports_file(cur_user, model)
    with open(reports_file, "w", encoding ='utf8') as f:
        json.dump(reports, f)

    results = {
        "status": "success",
    }
    if debug:
        print(results)
    return json.dumps(results)

########################################
# ROUTE: /SUBMIT_AVID_REPORT
@app.route("/submit_avid_report")
def submit_avid_report():
    cur_user = request.args.get("cur_user")
    name = request.args.get("name")
    email = request.args.get("email")
    reports_json = request.args.get("reports")

    reports = json.loads(reports_json)

    # Submit reports to AVID
    submit_reports_to_AVID(reports, cur_user, name, email)

    results = {
        "status": "success",
    }
    return json.dumps(results)

########################################
# ROUTE: /GET_EXPLORE_EXAMPLES
@app.route("/get_explore_examples")
def get_explore_examples():
    threshold = utils.get_toxic_threshold()
    n_examples = int(request.args.get("n_examples"))

    # Get sample of examples
    df = utils.get_explore_df(n_examples, threshold)
    ex_json = df.to_json(orient="records")

    results = {
        "examples": ex_json,
    }
    return json.dumps(results)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

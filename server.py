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

import audit_utils as utils

app = Flask(__name__)

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
comments_grouped_full_topic_cat = pd.read_pickle("data/comments_grouped_full_topic_cat2_persp.pkl")

@app.route("/audit_settings")
def audit_settings():
    # Fetch page content
    user = request.args.get("user")
    scaffold_method = request.args.get("scaffold_method")

    user_models = utils.get_all_model_names(user)
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

    if scaffold_method == "personal_cluster":
        cluster_model = user_models[0]
        personal_cluster_file = f"./data/personal_cluster_dfs/{cluster_model}.pkl"
        if os.path.isfile(personal_cluster_file) and cluster_model != "":
            print("audit_settings", personal_cluster_file, cluster_model)
            topics_under_top, topics_over_top = utils.get_personal_clusters(cluster_model)
            pers_cluster = topics_under_top + topics_over_top
            pers_cluster_options = {
                "label": "Personalized clusters",
                "options": [{"value": i, "text": cluster} for i, cluster in enumerate(pers_cluster)],
            }
            clusters_options.insert(0, pers_cluster_options)

    clusters_for_tuning = utils.get_large_clusters(min_n=150)
    clusters_for_tuning_options = [{"value": i, "text": cluster} for i, cluster in enumerate(clusters_for_tuning)]  # Format for Svelecte UI element

    context = {
        "personalized_models": user_models,
        "personalized_model_grp": grp_models,
        "perf_metrics": ["Average rating difference", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)"],
        "breakdown_categories": ['Topic', 'Toxicity Category', 'Toxicity Severity'],
        "clusters": clusters_options,
        "clusters_for_tuning": clusters_for_tuning_options,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_USERS
@app.route("/get_users")
def get_users():
    # Fetch page content
    with open(f"./data/users_to_models.pkl", "rb") as f:
        users_to_models = pickle.load(f)
        users = list(users_to_models.keys())
        context = {
            "users": users,
        }
        return json.dumps(context)

########################################
# ROUTE: /GET_AUDIT
@app.route("/get_audit")
def get_audit():
    pers_model = request.args.get("pers_model")
    perf_metric = request.args.get("perf_metric")
    breakdown_axis = request.args.get("breakdown_axis")
    breakdown_sort = request.args.get("breakdown_sort")
    n_topics = int(request.args.get("n_topics"))
    error_type = request.args.get("error_type")
    cur_user = request.args.get("cur_user")
    topic_vis_method = request.args.get("topic_vis_method") 
    if topic_vis_method == "null":
        topic_vis_method = "median"

    if breakdown_sort == "difference":
        sort_class_plot = True
    elif breakdown_sort == "default":
        sort_class_plot = False
    else:
        raise Exception("Invalid breakdown_sort value")

    overall_perf = utils.show_overall_perf(
        variant=pers_model,
        error_type=error_type,
        cur_user=cur_user,
        breakdown_axis=breakdown_axis,
        topic_vis_method=topic_vis_method,
    )

    results = {
        "overall_perf": overall_perf,
    }
    return json.dumps(results)

########################################
# ROUTE: /GET_CLUSTER_RESULTS
@app.route("/get_cluster_results")
def get_cluster_results():
    pers_model = request.args.get("pers_model")
    n_examples = int(request.args.get("n_examples"))
    cluster = request.args.get("cluster")
    example_sort = request.args.get("example_sort")
    comparison_group = request.args.get("comparison_group")
    topic_df_ids = request.args.getlist("topic_df_ids")
    topic_df_ids = [int(val) for val in topic_df_ids[0].split(",") if val != ""]
    search_type = request.args.get("search_type")
    keyword = request.args.get("keyword")
    n_neighbors = request.args.get("n_neighbors")
    if n_neighbors != "null":
        n_neighbors = int(n_neighbors)
    neighbor_threshold = 0.6
    error_type = request.args.get("error_type")
    use_model = request.args.get("use_model") == "true"
    scaffold_method = request.args.get("scaffold_method")
        

    # If user has a tuned model for this cluster, use that
    cluster_model_file = f"./data/trained_models/{pers_model}_{cluster}.pkl"
    if os.path.isfile(cluster_model_file):
        pers_model = f"{pers_model}_{cluster}"

    print(f"get_cluster_results using model {pers_model}")

    other_ids = []
    perf_metric = "avg_diff"
    sort_ascending = True if example_sort == "ascending" else False

    topic_df = None
    
    personal_cluster_file = f"./data/personal_cluster_dfs/{pers_model}.pkl"
    if (scaffold_method == "personal_cluster") and (os.path.isfile(personal_cluster_file)):
        # Handle personal clusters
        with open(personal_cluster_file, "rb") as f:
            topic_df = pickle.load(f)
            topic_df = topic_df[(topic_df["topic"] == cluster)]
    else:
        # Regular handling
        with open(f"data/preds_dfs/{pers_model}.pkl", "rb") as f:
            topic_df = pickle.load(f)
        if search_type == "cluster":
            # Display examples with comment, your pred, and other users' pred
            topic_df = topic_df[(topic_df["topic"] == cluster) | (topic_df["item_id"].isin(topic_df_ids))]
                
        elif search_type == "neighbors":
            neighbor_ids = utils.get_match(topic_df_ids, K=n_neighbors, threshold=neighbor_threshold, debug=False)
            topic_df = topic_df[(topic_df["item_id"].isin(neighbor_ids)) | (topic_df["item_id"].isin(topic_df_ids))]
        elif search_type == "keyword":
            topic_df = topic_df[(topic_df["comment"].str.contains(keyword, case=False, regex=False)) | (topic_df["item_id"].isin(topic_df_ids))]
    
    topic_df = topic_df.drop_duplicates()
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

    if (scaffold_method == "personal_cluster") and (os.path.isfile(personal_cluster_file)):
        cluster_overview_plot_json, sampled_df = utils.plot_overall_vis_cluster(topic_df, error_type=error_type, n_comments=500)
    else:
        # Regular
        cluster_overview_plot_json, sampled_df = utils.get_cluster_overview_plot(topic_df, error_type=error_type, use_model=use_model)

    cluster_comments = utils.get_cluster_comments(sampled_df,error_type=error_type, num_examples=n_examples, use_model=use_model)  # New version of cluster comment table

    results = {
        "topic_df_ids": topic_df_ids,
        "cluster_overview_plot_json": json.loads(cluster_overview_plot_json),
        "cluster_comments": cluster_comments, 
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
def get_group_model():
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
        comments_df=comments_grouped_full_topic_cat,
        n_label_per_bin=BIN_DISTRIB,
        score_bins=SCORE_BINS,
        grp_ids=grp_ids,
    )

    # print("ratings_grp", ratings_grp)

    # Modify model name
    model_name = f"{model_name}_group_gender{sel_gender}_relig{sel_relig}_pol{sel_pol}_race{sel_race_orig}_lgbtq_{sel_lgbtq}"

    label_dir = f"./data/labels/{model_name}"
    # Create directory for labels if it doesn't yet exist
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    last_label_i = len([name for name in os.listdir(label_dir) if (os.path.isfile(os.path.join(label_dir, name)) and name.endswith('.pkl'))])

    # Train group model
    mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, last_label_i, ratings_grp, user)

    duration = time.time() - start
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

    # model_name_suggestion = f"model_{int(time.time())}"
    model_name_suggestion = f"model_{user}"

    context = {
        "personalized_models": utils.get_all_model_names(user),
        "model_name_suggestion": model_name_suggestion,
        "clusters_for_tuning": clusters_for_tuning_options,
    }
    return json.dumps(context)

########################################
# ROUTE: /GET_COMMENTS_TO_LABEL
N_LABEL_PER_BIN = 8 # 8 * 5 = 40 comments
BIN_DISTRIB = [4, 8, 16, 8, 4]
SCORE_BINS = [(0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.01)]
@app.route("/get_comments_to_label")
def get_comments_to_label():
    n = int(request.args.get("n"))
    # Fetch examples to label
    to_label_ids = utils.create_example_sets(
        comments_df=comments_grouped_full_topic_cat,
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
N_LABEL_PER_BIN_TOPIC = 2 # 2 * 5 = 10 comments
@app.route("/get_comments_to_label_topic")
def get_comments_to_label_topic():
    # Fetch examples to label
    topic = request.args.get("topic")
    to_label_ids = utils.create_example_sets(
        comments_df=comments_grouped_full_topic_cat,
        # n_label_per_bin=N_LABEL_PER_BIN_TOPIC,
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
def get_personalized_model():
    model_name = request.args.get("model_name")
    ratings_json = request.args.get("ratings")
    mode = request.args.get("mode")
    user = request.args.get("user")
    ratings = json.loads(ratings_json)
    print(ratings)
    start = time.time()

    label_dir = f"./data/labels/{model_name}"
    # Create directory for labels if it doesn't yet exist
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    last_label_i = len([name for name in os.listdir(label_dir) if (os.path.isfile(os.path.join(label_dir, name)) and name.endswith('.pkl'))])

    # Handle existing or new model cases
    if mode == "view":
        # Fetch prior model performance
        if model_name not in utils.get_all_model_names():
            raise Exception(f"Model {model_name} does not exist")
        else:
            mae, mse, rmse, avg_diff, ratings_prev = utils.fetch_existing_data(model_name, last_label_i)
        
    elif mode == "train":
        # Train model and cache predictions using new labels
        print("get_personalized_model train")
        mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, last_label_i, ratings, user)
        
    duration = time.time() - start
    print("Time to train/cache:", duration) 

    perf_plot, mae_status = utils.plot_train_perf_results(model_name, mae)
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
        "duration": duration,
        "ratings_prev": ratings_prev,
        "perf_plot_json": json.loads(perf_plot_json),
    }
    return json.dumps(results)


########################################
# ROUTE: /GET_PERSONALIZED_MODEL_TOPIC
@app.route("/get_personalized_model_topic")
def get_personalized_model_topic():
    model_name = request.args.get("model_name")
    ratings_json = request.args.get("ratings")
    user = request.args.get("user")
    ratings = json.loads(ratings_json)
    topic = request.args.get("topic")
    print(ratings)
    start = time.time()

    # Modify model name
    model_name = f"{model_name}_{topic}"

    label_dir = f"./data/labels/{model_name}"
    # Create directory for labels if it doesn't yet exist
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    last_label_i = len([name for name in os.listdir(label_dir) if (os.path.isfile(os.path.join(label_dir, name)) and name.endswith('.pkl'))])

    # Handle existing or new model cases
    # Train model and cache predictions using new labels
    print("get_personalized_model_topic train")
    mae, mse, rmse, avg_diff, ratings_prev = utils.train_updated_model(model_name, last_label_i, ratings, user, topic=topic)
        
    duration = time.time() - start
    print("Time to train/cache:", duration) 

    def round_metric(x):
        return np.round(abs(x), 3)

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

    # Load reports for current user from stored files
    report_dir = f"./data/user_reports"
    user_file = os.path.join(report_dir, f"{cur_user}_{scaffold_method}.pkl")

    if not os.path.isfile(user_file):
        if scaffold_method == "fixed":
            reports = get_fixed_scaffold()
        elif (scaffold_method == "personal" or scaffold_method == "personal_group" or scaffold_method == "personal_test"):
            reports = get_personal_scaffold(model, topic_vis_method)
        elif (scaffold_method == "personal_cluster"):
            reports = get_personal_cluster_scaffold(model)
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
                    "complete_status": False,
                }
            ]
    else:
        # Load from pickle file
        with open(user_file, "rb") as f:
            reports = pickle.load(f)

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
            "complete_status": False,
        },
        {
            "title": "Topic: 73_troll_trolls_trolling_spammers",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 66_mexicans_mexico_mexican_spanish",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 89_cowards_coward_cowardly_brave",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Topic: 63_disgusting_gross_toxic_thicc",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
    ]

def get_empty_report(title, error_type):
    return {
            "title": f"Topic: {title}",
            "error_type": error_type,
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        }

def get_tutorial_scaffold():
    return [
        {
            "title": "Topic: 79_idiot_dumb_stupid_dumber",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
    ] 

def get_personal_cluster_scaffold(model):
    topics_under_top, topics_over_top = utils.get_personal_clusters(model)

    report_under = [get_empty_report(topic, "System is under-sensitive") for topic in topics_under_top]

    report_over = [get_empty_report(topic, "System is over-sensitive") for topic in topics_over_top]
    reports = (report_under + report_over)
    random.shuffle(reports)
    return reports

def get_topic_errors(df, topic_vis_method, threshold=2):
    topics = df["topic_"].unique().tolist()
    topic_errors = {}
    for topic in topics:
        t_df = df[df["topic_"] == topic]
        y_true = t_df["pred"].to_numpy()
        y_pred = t_df["rating"].to_numpy()
        if topic_vis_method == "mae":
            t_err = mean_absolute_error(y_true, y_pred)
        elif topic_vis_method == "mse":
            t_err = mean_squared_error(y_true, y_pred)
        elif topic_vis_method == "avg_diff":
            t_err = np.mean(y_true - y_pred)
        elif topic_vis_method == "fp_proportion":
            y_true = [0 if rating < threshold else 1 for rating in t_df["pred"].tolist()]
            y_pred = [0 if rating < threshold else 1 for rating in t_df["rating"].tolist()]
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except:
                tn, fp, fn, tp = [0, 0, 0, 0]  # ignore; set error to 0
            total = float(len(y_true))
            t_err = fp / total
        elif topic_vis_method == "fn_proportion":
            y_true = [0 if rating < threshold else 1 for rating in t_df["pred"].tolist()]
            y_pred = [0 if rating < threshold else 1 for rating in t_df["rating"].tolist()]
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except:
                tn, fp, fn, tp = [0, 0, 0, 0]  # ignore; set error to 0
            total = float(len(y_true))
            t_err = fn / total
        topic_errors[topic] = t_err
        
    return topic_errors

def get_personal_scaffold(model, topic_vis_method, n_topics=200, n=5):
    threshold = utils.get_toxic_threshold()

    # Get topics with greatest amount of error
    with open(f"./data/preds_dfs/{model}.pkl", "rb") as f:
        preds_df = pickle.load(f)
        preds_df_mod = preds_df.merge(utils.get_comments_grouped_full_topic_cat(), on="item_id", how="left", suffixes=('_', '_avg'))
        preds_df_mod = preds_df_mod[preds_df_mod["user_id"] == "A"].sort_values(by=["item_id"]).reset_index()
        preds_df_mod = preds_df_mod[preds_df_mod["topic_id_"] < n_topics]

        if topic_vis_method == "median":
            df = preds_df_mod.groupby(["topic_", "user_id"]).median().reset_index()
        elif topic_vis_method == "mean":
            df = preds_df_mod.groupby(["topic_", "user_id"]).mean().reset_index()
        elif topic_vis_method == "fp_fn":
            for error_type in ["fn_proportion", "fp_proportion"]:
                topic_errors = get_topic_errors(preds_df_mod, error_type)
                preds_df_mod[error_type] = [topic_errors[topic] for topic in preds_df_mod["topic_"].tolist()]
            df = preds_df_mod.groupby(["topic_", "user_id"]).mean().reset_index()
        else:
            # Get error for each topic
            topic_errors = get_topic_errors(preds_df_mod, topic_vis_method)
            preds_df_mod[topic_vis_method] = [topic_errors[topic] for topic in preds_df_mod["topic_"].tolist()]
            df = preds_df_mod.groupby(["topic_", "user_id"]).mean().reset_index()

        # Get system error
        df = df[(df["topic_"] != "53_maiareficco_kallystas_dyisisitmanila_tractorsazi") & (df["topic_"] != "79_idiot_dumb_stupid_dumber")]
        
        if topic_vis_method == "median" or topic_vis_method == "mean":
            df["error_magnitude"] = [utils.get_error_magnitude(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
            df["error_type"] = [utils.get_error_type_radio(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]

            df_under = df[df["error_type"] == "System is under-sensitive"]
            df_under = df_under.sort_values(by=["error_magnitude"], ascending=False).head(n) # surface largest errors first
            report_under = [get_empty_report(row["topic_"], row["error_type"]) for _, row in df_under.iterrows()]

            df_over = df[df["error_type"] == "System is over-sensitive"]
            df_over = df_over.sort_values(by=["error_magnitude"], ascending=False).head(n) # surface largest errors first
            report_over = [get_empty_report(row["topic_"], row["error_type"]) for _, row in df_over.iterrows()]
            
            # Set up reports
            # return [get_empty_report(row["topic_"], row["error_type"]) for index, row in df.iterrows()]
            reports = (report_under + report_over)
            random.shuffle(reports)
        elif topic_vis_method == "fp_fn":
            df_under = df.sort_values(by=["fn_proportion"], ascending=False).head(n)
            df_under = df_under[df_under["fn_proportion"] > 0]
            report_under = [get_empty_report(row["topic_"], "System is under-sensitive") for _, row in df_under.iterrows()]
            
            df_over = df.sort_values(by=["fp_proportion"], ascending=False).head(n)
            df_over = df_over[df_over["fp_proportion"] > 0]
            report_over = [get_empty_report(row["topic_"], "System is over-sensitive") for _, row in df_over.iterrows()]

            reports = (report_under + report_over)
            random.shuffle(reports)
        else:
            df = df.sort_values(by=[topic_vis_method], ascending=False).head(n * 2)
            df["error_type"] = [utils.get_error_type_radio(sys, user, threshold) for sys, user in zip(df["rating"].tolist(), df["pred"].tolist())]
            reports = [get_empty_report(row["topic_"], row["error_type"]) for _, row in df.iterrows()]

        return reports

def get_prompts_scaffold():
    return [
        {
            "title": "Are there terms that are used in your identity group or community that tend to be flagged incorrectly as toxic?",
            "error_type": "System is over-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Are there terms that are used in your identity group or community that tend to be flagged incorrectly as non-toxic?",
            "error_type": "System is under-sensitive",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Are there certain ways that your community tends to be targeted by outsiders?",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Are there other communities whose content should be very similar to your community's? Verify that this content is treated similarly by the system.",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
        {
            "title": "Are there ways that you've seen individuals in your community actively try to thwart the rules of automated content moderation systems? Check whether these strategies work here.",
            "error_type": "",
            "evidence": [],
            "text_entry": "",
            "complete_status": False,
        },
    ]

########################################
# ROUTE: /SAVE_REPORTS
@app.route("/save_reports")
def save_reports():
    cur_user = request.args.get("cur_user")
    reports_json = request.args.get("reports")
    reports = json.loads(reports_json)
    scaffold_method = request.args.get("scaffold_method")

    # Save reports for current user to stored files
    report_dir = f"./data/user_reports"
    # Save to pickle file
    with open(os.path.join(report_dir, f"{cur_user}_{scaffold_method}.pkl"), "wb") as f:
        pickle.dump(reports, f)

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
    df = utils.get_comments_grouped_full_topic_cat().sample(n=n_examples)

    df["system_decision"] = [utils.get_decision(rating, threshold) for rating in df["rating"].tolist()]
    df["system_color"] = [utils.get_user_color(sys, threshold) for sys in df["rating"].tolist()]  # get cell colors

    ex_json = df.to_json(orient="records")

    results = {
        "examples": ex_json,
    }
    return json.dumps(results)

########################################
# ROUTE: /GET_RESULTS
@app.route("/get_results")
def get_results():
    users = request.args.get("users")
    if users != "":
        users = users.split(",")
    # print("users", users)

    IGNORE_LIST = ["DemoUser"]
    report_dir = f"./data/user_reports"
    

    # For each user, get personal and prompt results
    # Get links to label pages and audit pages
    results = []
    for user in users:
        if user not in IGNORE_LIST:
            user_results = {}
            user_results["user"] = user
            for scaffold_method in ["personal", "personal_group", "prompts"]:
                # Get results
                user_file = os.path.join(report_dir, f"{user}_{scaffold_method}.pkl")
                if os.path.isfile(user_file):
                    with open(user_file, "rb") as f:
                        user_results[scaffold_method] = pickle.load(f)
            results.append(user_results)

    # print("results", results)

    results = {
        "results": results,
    }
    return json.dumps(results)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

import copy
import os
import numpy as np
import pandas as pd
import tabpfn
import pickle
from tabpfn.scripts.tabular_baselines import (
    gp_metric,
    knn_metric,
    xgb_metric,
    catboost_metric,
    transformer_metric,
    logistic_metric,
    autosklearn_metric,
    autosklearn2_metric,
    autogluon_metric,
    random_forest_metric,
)

from .data import get_data_split, get_X_y, load_all_data
from .run_llm_code import (
    convert_categorical_to_integer_f,
    create_mappings,
)

from .feature_extension_baselines import (
    extend_using_dfs,
    extend_using_autofeat,
    extend_using_caafe,
)


def evaluate_dataset(
    ds, df_train, df_test, prompt_id, name, method, metric_used, max_time=300, seed=0
):

    df_train = copy.deepcopy(df_train)
    if df_test is not None:
        df_test = copy.deepcopy(df_test)

    df_train = df_train.replace([np.inf, -np.inf], np.nan)

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train, df_test)

    # Apply the mappings to the train and test datasets
    non_target = [c for c in df_train.columns if c != ds[4][-1]]
    df_train[non_target] = df_train[non_target].apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df_test[non_target] = df_test[non_target].apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )

    df_train = df_train.astype(float)

    x, y = get_X_y(df_train, ds)

    if df_test is not None:
        # df_test = df_test.apply(lambda x: pd.factorize(x)[0])
        df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        try:
            df_test.loc[:, df_test.dtypes == object] = df_test.loc[
                :, df_test.dtypes == object
            ].apply(lambda x: pd.factorize(x)[0])
        except:
            pass
        df_test = df_test.astype(float)

        test_x, test_y = get_X_y(df_test, ds)

    np.random.seed(0)
    if method == "autogluon" or method == "autosklearn" or method == "autosklearn2":
        metric, ys, res = clf_dict[method](
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time
        )  #
    elif type(method) == str:
        metric, ys, res = clf_dict[method](
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time, no_tune={}
        )  #
    else:
        metric, ys, res = method(
            x, y, test_x, test_y, ds[3], metric_used, max_time=max_time
        )
    acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, ys)
    roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, ys)

    method_str = method if type(method) == str else "transformer"
    return {
        "acc": float(acc.numpy()),
        "roc": float(roc.numpy()),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }


def evaluate_dataset_helper_extend_df(
    df_train, df_test, ds, prompt_id, seed, code_overwrite=None
):
    # Remove target column from df_train
    target_train = df_train[ds[4][-1]]
    target_test = df_test[ds[4][-1]]
    df_train = df_train.drop(columns=[ds[4][-1]])
    df_test = df_test.drop(columns=[ds[4][-1]])

    if prompt_id == "dfs":
        df_train, df_test = extend_using_dfs(df_train, df_test, target_train)
    elif prompt_id == "autofeat":
        df_train, df_test = extend_using_autofeat(df_train, df_test, target_train)
    elif prompt_id == "v4" or prompt_id == "v3":
        df_train, df_test = extend_using_caafe(
            df_train, df_test, ds, seed, prompt_id, code_overwrite=code_overwrite
        )
    elif prompt_id == "v4+dfs" or prompt_id == "v3+dfs":
        df_train, df_test = extend_using_caafe(
            df_train, df_test, ds, seed, prompt_id[0:2]
        )
        df_train, df_test = extend_using_dfs(df_train, df_test, target_train)
    elif prompt_id == "v4+autofeat" or prompt_id == "v3+autofeat":
        df_train, df_test = extend_using_caafe(
            df_train, df_test, ds, seed, prompt_id[0:2]
        )
        df_train, df_test = extend_using_autofeat(df_train, df_test, target_train)

    # Add target column back to df_train
    df_train[ds[4][-1]] = target_train
    df_test[ds[4][-1]] = target_test
    # disable categorical encoding, because feature engineering might have broken indices
    ds[3] = []
    ds[2] = []

    return ds, df_train, df_test


def load_result(all_results, ds, seed, method, prompt_id="v2"):
    """Evaluates a dataframe with and without feature extension."""
    method_str = method if type(method) == str else "transformer"

    data_dir = os.environ.get("DATA_DIR", "data/")
    path = f"{data_dir}/evaluations/result_{ds[0]}_{prompt_id}_{seed}_{method_str}.txt"
    try:
        f = open(
            path,
            "rb",
        )
        r = pickle.load(f)
        f.close()
        r["failed"] = False

        all_results[f"{ds[0]}_{prompt_id}_{str(seed)}_{method_str}"] = r

        return r
    except Exception as e:
        try:
            path = f"{data_dir}/evaluations/result_{ds[0]}__{seed}_{method_str}.txt"
            f = open(
                path,
                "rb",
            )
            r = pickle.load(f)
            f.close()

            r["prompt"] = prompt_id
            r["failed"] = True

            all_results[f"{ds[0]}_{prompt_id}_{str(seed)}_{method_str}"] = r
            print(
                f"Could not load result for {ds[0]}_{prompt_id}_{str(seed)}_{method_str} {path}. BL loaded"
            )
            return r
        except Exception as e:
            print(
                f"[WARN] Could not load baseline result for {ds[0]}_{prompt_id}_{str(seed)}_{method_str} {path}"
            )
            return None


def evaluate_dataset_with_and_without_cafe(
    ds, seed, methods, metric_used, prompt_id="v2", max_time=300, overwrite=False
):
    """Evaluates a dataframe with and without feature extension."""
    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    ds, df_train, df_test = evaluate_dataset_helper_extend_df(
        df_train, df_test, ds, prompt_id, seed
    )

    print("SHAPE BEFORE", df_train_old.shape, "AFTER", df_train.shape)

    for method in methods:
        method_str = method if type(method) == str else "transformer"
        data_dir = os.environ.get("DATA_DIR", "data/")
        path = (
            f"{data_dir}/evaluations/result_{ds[0]}_{prompt_id}_{seed}_{method_str}.txt"
        )
        if os.path.exists(path) and not overwrite:
            print(f"Skipping {path}")
            continue
        print(ds[0], method_str, prompt_id, seed)
        r = evaluate_dataset(
            ds=ds,
            df_train=df_train,
            df_test=df_test,
            prompt_id=prompt_id,
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
            seed=seed,
        )
        f = open(
            path,
            "wb",
        )
        pickle.dump(r, f)
        f.close()


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        ds,
        df_train,
        df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)
        # ds_[4] = list(set(ds_[4]) - set([feat]))
        # ds_[3] = list(set(ds_[3]) - set([feat_idx]))

        res = evaluate_dataset(
            ds_,
            df_train_,
            df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances


clf_dict = {
    "gp": gp_metric,
    "knn": knn_metric,
    "catboost": catboost_metric,
    "xgb": xgb_metric,
    "transformer": transformer_metric,
    "logistic": logistic_metric,
    "autosklearn": autosklearn_metric,
    "autosklearn2": autosklearn2_metric,
    "autogluon": autogluon_metric,
    "random_forest": random_forest_metric,
}

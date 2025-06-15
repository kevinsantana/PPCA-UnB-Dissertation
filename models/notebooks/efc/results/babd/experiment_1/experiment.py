"""
Experiment 1: Multi-clas EFC dry run.
"""

import os

import pandas as pd
from babd.common.constants import LABELS_CM, METRICS_COLUMNS, SIZES_COLUMNS
from babd.common.format_dataset import run_babd_preprocessing_pipeline
from babd.experiment_1.dry_run import dry_run_babd
from loguru import logger

EXPERIMENT_FOLDER = "."
LABELS_CM.append("Technique")


def main(
    technique: str, fig_name: str, save_csv: bool = True, agg_features: bool = True
):
    try:
        extra = {
            "technique": technique,
            "fig_name": fig_name,
            "save_csv": save_csv,
            "agg_features": agg_features,
        }
        logger.bind(**extra).info("Initiating BABD dataset Experiment 1.")

        X_train, X_test, y_train, y_test = run_babd_preprocessing_pipeline()

        df_efc_sizes = pd.DataFrame(columns=SIZES_COLUMNS)
        df_efc_metrics = pd.DataFrame(columns=METRICS_COLUMNS)
        df_efc_confusion_matrix = pd.DataFrame(columns=LABELS_CM)

        technique_folder = technique
        fig_folder = os.path.join(EXPERIMENT_FOLDER, technique)
        fig_name_on_disk = fig_name.format(technique=technique_folder)

        extra["fig_folder"] = fig_folder
        extra["fig_name_on_disk"] = fig_name_on_disk
        logger.bind(**extra).info(
            f"Experiment params with technique {technique_folder}"
        )

        sizes, metrics, confusion_matrix = dry_run_babd(
            fig_folder=fig_folder,
            fig_name=fig_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            technique=technique_folder,
        )

        df_efc_sizes.loc[len(df_efc_sizes)] = sizes
        df_efc_metrics.loc[len(df_efc_metrics)] = metrics
        df_efc_confusion_matrix.loc[len(df_efc_confusion_matrix)] = confusion_matrix
        logger.bind(
            **{
                "sizes": sizes,
                "metrics": metrics,
                "confusion_matrix": confusion_matrix,
            }
        ).info("Experiment results")

        df_efc_metrics = df_efc_metrics.sort_values(by="f1_macro", ascending=False)
        logger.success("Finished BABD Experiment 1")

        if save_csv:
            logger.info(f"Saving experiments results to .csv to {fig_folder} folder.")
            df_efc_sizes.to_csv(
                f"{fig_folder}/df_efc_sizes.csv",
                sep=",",
                encoding="utf-8",
                index=False,
                header=True,
            )
            df_efc_metrics.to_csv(
                f"{fig_folder}/df_efc_metrics.csv",
                sep=",",
                encoding="utf-8",
                index=False,
                header=True,
            )
            df_efc_confusion_matrix.to_csv(
                f"{fig_folder}/df_efc_confusion_matrix.csv",
                sep=",",
                encoding="utf-8",
                index=False,
                header=True,
            )

        return df_efc_sizes, df_efc_metrics, df_efc_confusion_matrix

    except Exception as e:
        logger.bind(**extra).exception(
            f"Exception on experiment 1. Technique: {technique}. Exception: {str(e)}"
        )

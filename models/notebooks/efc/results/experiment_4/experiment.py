"""
Experiment 4
Applying feature selection to elliptic dataset.
"""
import os

import pandas as pd
from loguru import logger
from results.common.constants import LABELS_CM, METRICS_COLUMNS, SIZES_COLUMNS
from results.experiment_4.feature_selection_selectkbest_f_classif import (
    make_feature_selection_with_k_best,
    make_feature_selection_with_k_best_no_agg_features,
)

EXPERIMENT_FOLDER = "."
LABELS_CM.append("Technique")
K_SIZE = [10, 20, 30, 40, 50, 60]
ONLY_LABELED = True


def main(technique: str,
         fig_name: str,
         save_csv: bool = True,
         agg_features: bool = True):
    try:
        logger.info("Initiating Experiment 4")
        extra = {
            "technique": technique,
            "fig_name": fig_name,
            "save_csv": save_csv,
            "agg_features": agg_features
        }
        df_efc_sizes = pd.DataFrame(columns=SIZES_COLUMNS)
        df_efc_metrics = pd.DataFrame(columns=METRICS_COLUMNS)
        df_efc_confusion_matrix = pd.DataFrame(columns=LABELS_CM)

        for k in K_SIZE:
            technique_folder = technique.format(k_size=k)
            fig_folder = os.path.join(EXPERIMENT_FOLDER, technique)
            fig_name_on_disk = fig_name.format(technique=technique_folder)
            extra["fig_folder"] = fig_folder
            extra["fig_name_on_disk"] = fig_name_on_disk
            extra["k_size"] = k
            logger.bind(**extra).info(f"Experiment params with k_size={k} and technique {technique_folder}")
            if agg_features:
                sizes, metrics, confusion_matrix = make_feature_selection_with_k_best(
                    k=k,
                    fig_folder=fig_folder,
                    fig_name=fig_name_on_disk,
                    technique=technique_folder,
                )
            else:
                sizes, metrics, confusion_matrix = make_feature_selection_with_k_best_no_agg_features(
                    k=k,
                    fig_folder=fig_folder,
                    fig_name=fig_name_on_disk,
                    technique=technique_folder,
                )
            df_efc_sizes.loc[len(df_efc_sizes)] = sizes
            df_efc_metrics.loc[len(df_efc_metrics)] = metrics
            df_efc_confusion_matrix.loc[len(df_efc_confusion_matrix)] = confusion_matrix
            logger.bind(**{
                "sizes": sizes,
                "metrics": metrics,
                "confusion_matrix": confusion_matrix}
                ).info("Experiment results")

        df_efc_metrics = df_efc_metrics.sort_values(by='f1_macro', ascending=False)
        logger.success("Finished Experiment 4")

        if save_csv:
            logger.info(f"Saving experiments results to .csv to {fig_folder} folder.")
            df_efc_sizes.to_csv(f'{fig_folder}/df_efc_sizes.csv', sep=',', encoding='utf-8', index=False, header=True)
            df_efc_metrics.to_csv(f'{fig_folder}/df_efc_metrics.csv', sep=',', encoding='utf-8', index=False, header=True)
            df_efc_confusion_matrix.to_csv(f'{fig_folder}/df_efc_confusion_matrix.csv', sep=',', encoding='utf-8', index=False, header=True)

        return df_efc_sizes, df_efc_metrics, df_efc_confusion_matrix

    except Exception as e:
        logger.bind(**extra).exception(f"Exception on experiment 4. Technique: {technique}. Exception: {e}")

if __name__ == "__main__":
    main()

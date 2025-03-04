import os

import pandas as pd
from loguru import logger

from results.common.constants import METRICS_COLUMNS, SIZES_COLUMNS, LABELS_CM
from results.experiment_5.feature_selection_with_balanced_datasets import smote_with_feature_selection

EXPERIMENT_FOLDER = "results/experiment_5"
FIG_NAME = "{technique}_{k_size}"
LABELS_CM.append("Technique")
TECHNIQUE = "SMOTE With Feature Selection k={k_size}"
K_SIZE = [10, 20, 30, 40, 50, 60]
ONLY_LABELED = True


def main(save_csv: bool = True):
    try:
        logger.info("Initiating Experiment 5")
        for i, k in enumerate(K_SIZE):
            logger.info(f"Running with k_size={k}")
            FIG_FOLDER = os.path.join(EXPERIMENT_FOLDER, "1_smote")
            df_efc_sizes = pd.DataFrame(columns=SIZES_COLUMNS)
            df_efc_metrics = pd.DataFrame(columns=METRICS_COLUMNS)
            df_efc_confusion_matrix = pd.DataFrame(columns=LABELS_CM)
            FIG_NAME_k_10 = FIG_NAME.format(technique="1_smote_with_feature_selection_score_function_f_classif_k", k_size=k)
            sizes, metrics, confusion_matrix_values = smote_with_feature_selection(
            technique=TECHNIQUE.format(k_size=k),
            fig_folder=FIG_FOLDER,
            fig_name=FIG_NAME_k_10,
            k=k,
            only_labeled=ONLY_LABELED)
            df_efc_sizes.loc[i] = sizes
            df_efc_metrics[i] = metrics
            df_efc_confusion_matrix[i] = confusion_matrix_values
        logger.success("Finished Experiment 5")
        if save_csv:
            df_efc_sizes.to_csv(f'{EXPERIMENT_FOLDER}/df_efc_sizes.csv', sep=',', encoding='utf-8', index=False, header=True)
            df_efc_metrics.to_csv(f'{EXPERIMENT_FOLDER}/df_efc_metrics.csv', sep=',', encoding='utf-8', index=False, header=True)
            df_efc_confusion_matrix.to_csv(f'{EXPERIMENT_FOLDER}/df_efc_confusion_matrix.csv', sep=',', encoding='utf-8', index=False, header=True)

        return df_efc_sizes, df_efc_metrics, df_efc_confusion_matrix
    except Exception as e:
        logger.error(f" Exception on Experiment 5: {str(e)}")


if __name__ == "__main__":
    main()

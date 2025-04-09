import os

import pandas as pd
from loguru import logger
from results.common.constants import LABELS_CM, METRICS_COLUMNS, SIZES_COLUMNS
from results.experiment_5.feature_selection_with_balanced_datasets import (
    smote_with_feature_selection,
)

EXPERIMENT_FOLDER = "."
LABELS_CM.append("Technique")
K_SIZE = [10, 20, 30, 40, 50, 60]
ONLY_LABELED = True


def main(
    technique: str,
    fig_name: str,
    save_csv: bool = True,
):
    try:
        logger.info("Initiating Experiment 5")
        df_efc_sizes = pd.DataFrame(columns=SIZES_COLUMNS)
        df_efc_metrics = pd.DataFrame(columns=METRICS_COLUMNS)
        df_efc_confusion_matrix = pd.DataFrame(columns=LABELS_CM)

        for k in K_SIZE:
            technique_folder = technique.format(k_size=k)
            fig_folder = os.path.join(
                EXPERIMENT_FOLDER,
                technique
                or "1_smote_with_feature_selection_score_function_f_classif_k",
            )
            fig_name_on_disk = fig_name.format(technique=technique_folder)
            extra = {
                "technique": technique_folder,
                "k_size": k,
                "fig_name": fig_name_on_disk,
                "fig_folder": fig_folder,
                "only_labeled": ONLY_LABELED,
            }
            logger.bind(**extra).info(
                f"Experiment params with k_size={k} and technique {technique_folder}"
            )
            sizes, metrics, confusion_matrix_values = smote_with_feature_selection(
                technique=technique_folder,
                fig_folder=fig_folder,
                fig_name=fig_name_on_disk,
                k=k,
                only_labeled=ONLY_LABELED,
            )
            df_efc_sizes.loc[len(df_efc_sizes)] = sizes
            df_efc_metrics.loc[len(df_efc_metrics)] = metrics
            df_efc_confusion_matrix.loc[len(df_efc_confusion_matrix)] = (
                confusion_matrix_values
            )
            logger.bind(
                **{
                    "sizes": sizes,
                    "metrics": metrics,
                    "confusion_matrix_values": confusion_matrix_values,
                }
            ).info("SMOTE results")

        df_efc_metrics = df_efc_metrics.sort_values(by="f1_macro", ascending=False)
        logger.success("Finished Experiment 5")

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
        logger.exception(f"Exception on Experiment 5: {str(e)}")


if __name__ == "__main__":
    main()

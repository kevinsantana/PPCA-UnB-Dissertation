import csv
import itertools
import logging
import sys

import pandas as pd


def setup_logging():
    """Return a logger that sends message to stdout."""
    logging.basicConfig(
        format="%(asctime)s, %(name)s (%(filename)s:%(lineno)s) %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logging()


def main():
    try:
        be_file_path = "/Users/kevinaraujo/repos/dissertation/PPCA-UnB-Dissertation/models/notebooks/efc/datasets/elliptic2/background_edges.csv"
        bn_file_path = "/Users/kevinaraujo/repos/dissertation/PPCA-UnB-Dissertation/models/notebooks/efc/datasets/elliptic2/background_nodes.csv"
        nodes_file_path = "/Users/kevinaraujo/repos/dissertation/PPCA-UnB-Dissertation/models/notebooks/efc/datasets/elliptic2/nodes.csv"
        cc_file_path = "/Users/kevinaraujo/repos/dissertation/PPCA-UnB-Dissertation/models/notebooks/efc/datasets/elliptic2/connected_components.csv"
        be_csv_chunked, bn_csv_chunked, nodes_csv_chunked, cc_csv_chunked = (
            _chunk_elliptic2_data(
                be_file_path=be_file_path,
                bn_file_path=bn_file_path,
                nodes_file_path=nodes_file_path,
                cc_file_path=cc_file_path,
            )
        )
        row_generator = _read_and_merge_chunks(
            be_csv_chunked=be_csv_chunked,
            bn_csv_chunked=bn_csv_chunked,
            nodes_csv_chunked=nodes_csv_chunked,
            cc_csv_chunked=cc_csv_chunked,
        )
        df = _create_merged_df_sample(row_generator=row_generator)
        return df

    except Exception as e:
        logger.exception("Unexpected exception occured: %s", str(e))


def _chunk_elliptic2_data(
    be_file_path: str,
    bn_file_path: str,
    nodes_file_path: str,
    cc_file_path: str,
    chunk_size: int = 1_000_000,
):
    try:
        with open(be_file_path, mode="r", encoding="utf-8") as be_csv, open(
            bn_file_path, mode="r", encoding="utf-8"
        ) as bn_csv, open(
            nodes_file_path, mode="r", encoding="utf-8"
        ) as nodes_csv, open(
            cc_file_path, mode="r", encoding="utf-8"
        ) as cc_csv:
            be_csv_reader = csv.DictReader(be_csv)
            bn_csv_reader = csv.DictReader(bn_csv)
            nodes_csv_reader = csv.DictReader(nodes_csv)
            cc_csv_reader = csv.DictReader(cc_csv)

            be_csv_chunked_list = list(itertools.islice(be_csv_reader, chunk_size))
            be_csv_chunked_dict = {}
            bn_csv_chunked_list = list(itertools.islice(bn_csv_reader, chunk_size))
            bn_csv_chunked_dict = {}
            nodes_csv_chunked_list = list(
                itertools.islice(nodes_csv_reader, chunk_size)
            )
            nodes_csv_chunked_dict = {}
            cc_csv_chunked_list = list(itertools.islice(cc_csv_reader, chunk_size))
            cc_csv_chunked_dict = {}

            for be_csv in be_csv_chunked_list:
                be_csv_chunked_dict[be_csv["clId2"]] = be_csv

            for bn_csv in bn_csv_chunked_list:
                bn_csv_chunked_dict[bn_csv["clId"]] = bn_csv

            for nodes_csv in nodes_csv_chunked_list:
                nodes_csv_chunked_dict[nodes_csv["clId"]] = nodes_csv

            for cc_csv in cc_csv_chunked_list:
                cc_csv_chunked_dict[cc_csv["ccId"]] = cc_csv

            return (
                be_csv_chunked_dict,
                bn_csv_chunked_dict,
                nodes_csv_chunked_dict,
                cc_csv_chunked_dict,
            )
            # return be_csv_chunked_list, bn_csv_chunked_list, nodes_csv_chunked_list, cc_csv_chunked_list

    except Exception as e:
        logger.error(
            "An error occured while chunking the elliptic2 csv data %s.", str(e)
        )
        raise


def _read_and_merge_chunks(
    be_csv_chunked, bn_csv_chunked, nodes_csv_chunked, cc_csv_chunked
):
    """
    Processes CSV data from iterators yielding dictionaries, merges them row by row,
    and yields merged rows as dictionaries.
    Assumes input iterators (be_csv_chunked, etc.) yield dictionaries directly
    (e.g., from csv.DictReader wrapped by itertools.islice).
    """
    try:
        edge_feat_map = {f"feat#{i}": f"edge_feat_{i}" for i in range(1, 96)}
        node_feat_map = {f"feat#{i}": f"node_feat_{i}" for i in range(1, 44)}

        processed_rows_count = 0
        # Each of d_be, d_bn, d_n, d_cc is a dictionary from the respective DictReader chunk.
        for d_be, d_bn, d_n, d_cc in zip(
            be_csv_chunked.values(),
            bn_csv_chunked.values(),
            nodes_csv_chunked.values(),
            cc_csv_chunked.values(),
        ):
            merged_row = {}

            # 1. From nodes.csv (n) - canonical clId, ccId, label
            merged_row["clId"] = d_n.get("clId")
            merged_row["ccId"] = d_n.get("ccId")
            merged_row["label"] = d_n.get("label")
            # Add other features from nodes.csv, prefixed
            for k_n, v_n in d_n.items():
                if k_n not in ["clId", "ccId", "label"]:
                    merged_row[f"n_{k_n}"] = v_n

            # Verification for linking keys
            valid_link = True
            if not (d_n.get("ccId") == d_cc.get("ccId")):
                logger.debug(
                    f"ccId mismatch: nodes ({d_n.get('ccId')}) vs cc ({d_cc.get('ccId')}) for clId {d_n.get('clId')}"
                )
                valid_link = False
            if not (d_n.get("clId") == d_bn.get("clId")):
                logger.debug(
                    f"clId mismatch: nodes ({d_n.get('clId')}) vs bn ({d_bn.get('clId')})"
                )
                valid_link = False
            # ccId might also be in d_bn, check consistency if it exists
            if "ccId" in d_bn and d_n.get("ccId") != d_bn.get("ccId"):
                logger.debug(
                    f"ccId mismatch: nodes ({d_n.get('ccId')}) vs bn ({d_bn.get('ccId')}) for clId {d_n.get('clId')}"
                )
                valid_link = False
            if not (d_n.get("clId") == d_be.get("clId2")):
                logger.debug(
                    f"clId mismatch: nodes ({d_n.get('clId')}) vs be_clId2 ({d_be.get('clId2')})"
                )
                valid_link = False

            if not valid_link:
                logger.warning(
                    f"Skipping row due to key mismatch for primary clId: {d_n.get('clId')}"
                )
                continue

            # 2. From connected_components.csv (cc)
            merged_row["ccLabel"] = d_cc.get("ccLabel")

            # 3. From background_nodes.csv (bn)
            for h_bn_orig, val_bn in d_bn.items():
                if h_bn_orig in node_feat_map:
                    merged_row[node_feat_map[h_bn_orig]] = val_bn
                elif h_bn_orig not in ["clId", "ccId"]:  # Avoid re-adding linking keys
                    # Prefix other columns to prevent clashes if they exist in nodes.csv
                    merged_row[f"bn_{h_bn_orig}"] = val_bn

            # 4. From background_edges.csv (be)
            merged_row["clId1"] = d_be.get("clId1")  # This is txId1
            for h_be_orig, val_be in d_be.items():
                if h_be_orig in edge_feat_map:
                    merged_row[edge_feat_map[h_be_orig]] = val_be
                elif h_be_orig not in [
                    "clId1",
                    "clId2",
                ]:  # Avoid re-adding linking/key columns
                    # Prefix other columns
                    merged_row[f"be_{h_be_orig}"] = val_be

            yield merged_row
            processed_rows_count += 1
            if processed_rows_count % 50000 == 0:
                logger.info(f"Processed {processed_rows_count} rows...")

        logger.info(
            f"Finished processing_csv_files. Total rows merged: {processed_rows_count}"
        )

    except Exception as e:
        logger.error(
            "An unexpected error occured while processing csv chunk data: %s", str(e)
        )
        raise


def _create_merged_df_sample(row_generator):
    try:
        rows_for_gcs_part = []
        # Max size for a part, e.g., 10 GB. GCS has a 5TB object limit, but 10GB is more manageable.
        MAX_PART_BYTES = 10 * 1024 * 1024 * 1024
        # Check size less frequently to avoid overhead of df.to_csv() too often
        ROW_CHECK_INTERVAL = 100_000  # Check CSV size estimate every X rows

        logger.info(
            f"Starting to accumulate rows for GCS upload. Target part size: ~{MAX_PART_BYTES / (1024*1024*1024)} GB."
        )

        for i, merged_row_dict in enumerate(row_generator):
            rows_for_gcs_part.append(merged_row_dict)
            if (i + 1) % ROW_CHECK_INTERVAL == 0:
                if not rows_for_gcs_part:
                    continue

        df_current_part = pd.DataFrame(rows_for_gcs_part)
        logger.info(
            "Successfully created a merged dataframe from elliptic2 dataset sample."
        )

        return df_current_part

    except Exception as e:
        logger.error(
            "An unexpected error occured in %s. Exception: %s", __name__, str(e)
        )
        raise


if __name__ == "__main__":
    df = main()
    print(df)

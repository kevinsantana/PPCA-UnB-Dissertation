"""
These functions should run on a server, so we can read the entire files in memory.
As we can't assure the files are someway sorted.
"""

import csv
import logging
import sys

import pandas as pd
from google.cloud import storage


def setup_logging():
    """Return a logger that sends message to stdout."""
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s (%(filename)s:%(lineno)s) %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logging()


def process_csv_files(
    background_edges_lines,
    background_nodes_lines,
    nodes_lines,
    connected_components_lines,
    delimiter=",",
):
    """
    Processes CSV data from in-memory line iterables, merges them row by row,
    and yields merged rows as dictionaries.

    Assumes all input iterables (after headers) have the same number of rows
    and are ordered consistently for a direct row-wise merge based on linking logic.

    Column renaming strategy:
    - Features from background_edges (feat#X) -> edge_feat_X
    - Features from background_nodes (feat#X) -> node_feat_X
    - Canonical clId, ccId, label come from nodes_lines.
    - ccLabel comes from connected_components_lines.
    - Other potentially clashing columns from background_edges/nodes are prefixed.
    """
    logger.info("Starting to process and merge CSV data row by row.")

    try:
        be_reader = csv.reader(background_edges_lines, delimiter=delimiter)
        bn_reader = csv.reader(background_nodes_lines, delimiter=delimiter)
        n_reader = csv.reader(nodes_lines, delimiter=delimiter)
        cc_reader = csv.reader(connected_components_lines, delimiter=delimiter)

        # Read headers
        h_be = next(be_reader, None)
        h_bn = next(bn_reader, None)
        h_n = next(n_reader, None)
        h_cc = next(cc_reader, None)

        if not (h_be and h_bn and h_n and h_cc):
            logger.warning(
                "Error: One or more CSV data streams are empty or missing headers."
            )
            return

        edge_feat_map = {f"feat#{i}": f"edge_feat_{i}" for i in range(1, 96)}
        node_feat_map = {f"feat#{i}": f"node_feat_{i}" for i in range(1, 44)}

        processed_rows_count = 0
        for r_be, r_bn, r_n, r_cc in zip(be_reader, bn_reader, n_reader, cc_reader):
            if not (
                r_be and r_bn and r_n and r_cc
            ):  # Should not happen if lengths are same
                logger.warning("Row data missing from one of the files. Stopping.")
                break

            d_be = dict(zip(h_be, r_be))
            d_bn = dict(zip(h_bn, r_bn))
            d_n = dict(zip(h_n, r_n))
            d_cc = dict(zip(h_cc, r_cc))

            merged_row = {}

            # 1. From nodes.csv (n) - canonical clId, ccId, label
            merged_row["clId"] = d_n.get("clId")
            merged_row["ccId"] = d_n.get("ccId")
            merged_row["label"] = d_n.get("label")
            for k_n, v_n in d_n.items():  # Add other features from nodes.csv, prefixed
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
            f"An unexpected error occurred during CSV processing: {e}", exc_info=True
        )
        raise


def download_and_process_from_gcs(
    bucket_name,
    background_edges_blob_name,
    background_nodes_blob_name,
    connected_components_blob_name,
    nodes_blob_name,
    delimiter=",",
):
    """
    Downloads CSV files from GCS, merges them, and uploads to GCS in 10GB parts.

    Args:
        bucket_name (str): The name of the GCS bucket.
        background_edges_blob_name (str): The path to the edges CSV file in the bucket.
        background_nodes_blob_name (str): The path to the nodes CSV file in the bucket.
        connected_components_blob_name (str): The path to the components CSV file in the bucket.
        nodes_blob_name(str): The path to the nodes CSV file in the bucket.
        delimiter (str): The delimiter used in the CSV files.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    processed_bucket = storage_client.bucket(
        bucket_name
    )  # Assuming same bucket for output

    try:
        logger.info(
            f"Downloading {background_edges_blob_name} from GCS bucket {bucket_name} into memory..."
        )
        edges_blob = bucket.blob(background_edges_blob_name)
        edges_content_str = edges_blob.download_as_text(encoding="utf-8")
        edges_lines = edges_content_str.splitlines()
        logger.info(
            f"Finished downloading {background_edges_blob_name} into memory ({len(edges_lines)} lines)."
        )

        logger.info(
            f"Downloading {background_nodes_blob_name} from GCS bucket {bucket_name} into memory..."
        )
        background_nodes_blob = bucket.blob(background_nodes_blob_name)
        background_nodes_content_str = background_nodes_blob.download_as_text(
            encoding="utf-8"
        )
        background_nodes_lines = background_nodes_content_str.splitlines()
        logger.info(
            f"Finished downloading {background_nodes_blob_name} into memory ({len(background_nodes_lines)} lines)."
        )

        logger.info(
            f"Downloading {nodes_blob_name} from GCS bucket {bucket_name} into memory..."
        )
        nodes_blob = bucket.blob(nodes_blob_name)
        nodes_content_str = nodes_blob.download_as_text(encoding="utf-8")
        nodes_lines = nodes_content_str.splitlines()
        logger.info(
            f"Finished downloading {nodes_blob_name} into memory ({len(nodes_lines)} lines)."
        )

        logger.info(
            f"Downloading {connected_components_blob_name} from GCS bucket {bucket_name} into memory..."
        )
        components_blob = bucket.blob(connected_components_blob_name)
        components_content_str = components_blob.download_as_text(encoding="utf-8")
        components_lines = components_content_str.splitlines()
        logger.info(
            f"Finished downloading {connected_components_blob_name} into memory ({len(components_lines)} lines)."
        )

        # Determine total expected data rows (assuming edges_lines is representative and includes a header)
        total_data_rows = len(edges_lines) - 1 if edges_lines else 0
        if total_data_rows <= 0:
            logger.warning("No data rows to process after downloading files.")
            return

        row_generator = process_csv_files(
            edges_lines,
            background_nodes_lines,
            nodes_lines,
            components_lines,
            delimiter=delimiter,
        )

        rows_for_gcs_part = []
        current_estimated_csv_size = 0
        part_counter = 0
        # Max size for a part, e.g., 10 GB. GCS has a 5TB object limit, but 10GB is more manageable.
        MAX_PART_BYTES = 10 * 1024 * 1024 * 1024
        # Check size less frequently to avoid overhead of df.to_csv() too often
        ROW_CHECK_INTERVAL = 100_000  # Check CSV size estimate every X rows

        logger.info(
            f"Starting to accumulate rows for GCS upload. Target part size: ~{MAX_PART_BYTES / (1024*1024*1024)} GB."
        )

        for i, merged_row_dict in enumerate(row_generator):
            rows_for_gcs_part.append(merged_row_dict)

            # Periodically check size or if it's the last row
            if (i + 1) % ROW_CHECK_INTERVAL == 0 or (i + 1) == total_data_rows:
                if not rows_for_gcs_part:
                    continue

                df_current_part = pd.DataFrame(rows_for_gcs_part)
                # Estimate CSV size by actually converting (can be slow for very large intermediate DFs)
                # A more performant way might involve sampling or average row size if this becomes a bottleneck.
                try:
                    csv_string_for_size_check = df_current_part.to_csv(
                        index=False
                    ).encode("utf-8")
                    current_estimated_csv_size = len(csv_string_for_size_check)
                    del csv_string_for_size_check  # Free memory
                except MemoryError:
                    logger.warning(
                        f"MemoryError during CSV size estimation with {len(rows_for_gcs_part)} rows. Estimating based on DataFrame size."
                    )
                    current_estimated_csv_size = (
                        sys.getsizeof(df_current_part) * 2
                    )  # Rough heuristic

                logger.info(
                    f"Accumulated {len(rows_for_gcs_part)} rows. Estimated CSV size: {current_estimated_csv_size / (1024*1024):.2f} MB."
                )

                if (
                    current_estimated_csv_size >= MAX_PART_BYTES
                    or (i + 1) == total_data_rows
                ):
                    part_counter += 1
                    csv_data_to_upload = df_current_part.to_csv(index=False)

                    blob_name = f"elliptic2/processed/elliptic2_{part_counter}.csv"
                    blob = processed_bucket.blob(blob_name)
                    logger.info(
                        f"Uploading part {part_counter} ({len(csv_data_to_upload)/(1024*1024):.2f} MB) to GCS: gs://{bucket_name}/{blob_name}"
                    )
                    blob.upload_from_string(csv_data_to_upload, content_type="text/csv")
                    logger.info(f"Successfully uploaded {blob_name}")

                    rows_for_gcs_part = []
                    current_estimated_csv_size = 0
                    del df_current_part  # Free memory

        logger.info(f"All parts uploaded. Total parts: {part_counter}")

    except Exception as e:
        logger.error(f"An error occurred during GCS download: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Processing CSV files from GCS bucket (direct to memory)...")
    # This will now perform uploads directly, not yield data for the loop below.
    download_and_process_from_gcs(
        bucket_name="masters-degree-datasets",  # Replace with your bucket name
        background_edges_blob_name="elliptic2/background_edges.csv",  # Corrected typo if any
        background_nodes_blob_name="elliptic2/background_nodes.csv",  # Replace with your blob name
        nodes_blob_name="elliptic2/nodes.csv",  # Placeholder: Ensure this file exists or adjust path
        connected_components_blob_name="elliptic2/connected_components.csv",  # Replace with your blob name
        delimiter=",",
    )
    logger.info("Script finished.")

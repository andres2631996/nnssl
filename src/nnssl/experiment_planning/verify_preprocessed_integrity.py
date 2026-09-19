"""
Standalone integrity check for preprocessed .b2nd / .pkl files.

`nnSSLDatasetBlosc2.save_case` (nnssl/data/dataloading/dataset.py) writes each array
directly to its final path via `blosc2.asarray(..., urlpath=<final path>, mmap_mode="w+")`.
This is not atomic: if a preprocessing worker is OOM-killed or a node is preempted
mid-write (more likely the larger the image, since the write simply takes longer),
the final `.b2nd` file is left truncated on disk. `os.path.exists` (used both by the
preprocessor's "already done, skip" check and by the training-time dataset loader)
cannot tell a truncated file from a valid one, so the corruption stays invisible until
a training worker tries to `blosc2.open()` it, potentially crashing/hanging mid-epoch.

This module scans a preprocessed dataset/configuration folder, tries to actually open
(and, with `full_check=True`, fully decompress) every `.b2nd` file and load every
`.pkl` file, and reports/optionally deletes anything that fails. Deleting a corrupt
file makes the preprocessor's existence-based skip-check see that case as incomplete
again, so the next `nnssl_preprocess` run regenerates it.
"""

import glob
import os
from dataclasses import dataclass
from multiprocessing import get_context
from typing import List, Optional, Tuple

import blosc2
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.paths import nnssl_preprocessed
from nnssl.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


@dataclass
class CorruptionReport:
    path: str
    kind: str  # "b2nd" or "pkl"
    error: str


def _check_single_b2nd(args: Tuple[str, bool]) -> Optional[CorruptionReport]:
    filepath, full_check = args
    try:
        arr = blosc2.open(urlpath=filepath, mode="r", dparams={"nthreads": 1})
        _ = arr.shape  # forces parsing of the file header/chunk index
        if full_check:
            _ = arr[...]  # forces decompression of every chunk, not just the header
        del arr
    except Exception as e:
        print(filepath)
        return CorruptionReport(
            path=filepath, kind="b2nd", error=f"{type(e).__name__}: {e}"
        )


def _check_single_pkl(filepath: str) -> Optional[CorruptionReport]:
    try:
        load_pickle(filepath)
    except Exception as e:
        print(filepath)
        return CorruptionReport(
            path=filepath, kind="pkl", error=f"{type(e).__name__}: {e}"
        )


def verify_preprocessed_folder(
    root_dir: str,
    full_check: bool = False,
    num_processes: int = 1,
    delete_corrupt: bool = False,
) -> List[CorruptionReport]:
    """
    Recursively scans `root_dir` (e.g. nnssl_preprocessed/<Dataset>/<data_identifier>)
    for .b2nd and .pkl files and verifies each one is actually openable.
    """
    b2nd_files = sorted(glob.glob(join(root_dir, "**", "*.b2nd"), recursive=True))
    # A ".tmp.b2nd" is a leftover from an interrupted atomic write and is expected to be partial.
    b2nd_files = [f for f in b2nd_files if not f.endswith(".tmp.b2nd")]
    pkl_files = sorted(glob.glob(join(root_dir, "**", "*.pkl"), recursive=True))

    print(f"  Found {len(b2nd_files)} .b2nd files and {len(pkl_files)} .pkl files.")

    if num_processes > 1 and (b2nd_files or pkl_files):
        ctx = get_context("spawn")
        with ctx.Pool(num_processes) as p:
            b2nd_results = p.map(
                _check_single_b2nd, [(f, full_check) for f in b2nd_files]
            )
            pkl_results = p.map(_check_single_pkl, pkl_files)
    else:
        for f in b2nd_files:
            _ = _check_single_b2nd((f, False))
        for f in pkl_files:
            _ = _check_single_pkl(f)
        # b2nd_results = [_check_single_b2nd((f, full_check)) for f in b2nd_files]
        # pkl_results = [_check_single_pkl(f) for f in pkl_files]


def verify_preprocessed_dataset(
    dataset_ids: List[int],
    plans_identifier: str = "nnsslPlans",
    configurations: Optional[List[str]] = None,
    full_check: bool = False,
    num_processes: int = 1,
    delete_corrupt: bool = False,
) -> List[CorruptionReport]:
    """
    Verifies the preprocessed output of one or more datasets/configurations.
    `configurations=None` checks every configuration listed in the plans file.
    """
    all_reports: List[CorruptionReport] = []
    for dataset_id in dataset_ids:
        dataset_name = convert_id_to_dataset_name(dataset_id)
        plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
        if not os.path.isfile(plans_file):
            print(
                f"INFO: Plans file {plans_file} not found. Skipping dataset {dataset_name}."
            )
            continue
        plan = Plan.load_from_file(plans_file)
        configs_to_check = (
            configurations
            if configurations is not None
            else list(plan.configurations.keys())
        )

        for c in configs_to_check:
            if c not in plan.configurations:
                print(
                    f"INFO: Configuration {c} not found in plans file {plans_identifier}.json "
                    f"of dataset {dataset_name}. Skipping."
                )
                continue
            config_plan = plan.configurations[c]
            root_dir = join(
                nnssl_preprocessed, dataset_name, config_plan.data_identifier
            )
            if not os.path.isdir(root_dir):
                print(f"INFO: {root_dir} does not exist yet. Skipping.")
                continue
            print(f"Verifying {dataset_name} / {c} ({root_dir}) ...")
            reports = verify_preprocessed_folder(
                root_dir,
                full_check=full_check,
                num_processes=num_processes,
                delete_corrupt=delete_corrupt,
            )
            all_reports.extend(reports)

    print(
        f"\nTotal corrupt/unreadable files across all requested datasets/configurations: {len(all_reports)}"
    )
    if delete_corrupt and all_reports:
        print(
            "Corrupt files were deleted. Re-run `nnssl_preprocess` for the affected dataset(s)/"
            "configuration(s) to regenerate those cases."
        )
    return all_reports

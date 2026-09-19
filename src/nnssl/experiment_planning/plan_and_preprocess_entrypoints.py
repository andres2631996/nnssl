from time import sleep
from nnssl.configuration import default_num_processes
from nnssl.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)
from nnssl.experiment_planning.verify_preprocessed_integrity import (
    verify_preprocessed_dataset,
)
from nnssl.preprocessing.preprocessors.default_preprocessor import (
    PREPROCESS_SPACING_STYLES,
)
from typing import get_args
from loguru import logger


def extract_fingerprint_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-np",
        type=int,
        default=default_num_processes,
        required=False,
        help=f"[OPTIONAL] Number of processes used for fingerprint extraction. "
        f"Default: {default_num_processes}",
    )
    parser.add_argument(
        "--clean",
        required=False,
        default=False,
        action="store_true",
        help="[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a "
        "fingerprint already exists, the fingerprint extractor will not run.",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints([args.d], args.np, args.clean, args.verbose)


def plan_experiment_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        nargs="+",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-pl",
        type=str,
        default="ExperimentPlanner",
        required=False,
        help="[OPTIONAL] Name of the Experiment Planner class that should be used. Default is "
        "'ExperimentPlanner'. Note: There is no longer a distinction between 2d and 3d planner. "
        "It's an all in one solution now. Wuch. Such amazing.",
    )
    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(
        args.d,
        args.pl,
    )


def preprocess_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        nargs="+",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-c",
        nargs="+",
        type=str,
        default=["onemmiso"],
        help="List of configurations. This will create a pre-training dataset that is resampled in the respective way.",
        choices=get_args(PREPROCESS_SPACING_STYLES),
    )
    parser.add_argument(
        "-plans_name",
        default="nnsslPlans",
        required=False,
        help="[OPTIONAL] You can use this to specify a custom plans file that you may have generated",
    )
    parser.add_argument(
        "-part",
        type=int,
        default=0,
        required=False,
        help="[OPTIONAL] Defines which of the evenly sized chunks to process. Must be < `-total_parts`",
    )
    parser.add_argument(
        "-total_parts",
        type=int,
        default=1,
        required=False,
        help="[OPTIONAL] This is used for parallelization. Allows to split preprocessing in non-overlapping evenly chunked parts.",
    )
    parser.add_argument(
        "-np",
        type=int,
        nargs="+",
        default=[8, 4, 8],
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with -c. If it's a "
        "list of numbers this list must have as many elements as there are configurations. We "
        "then iterate over zip(configs, num_processes) to determine then umber of processes "
        "used for each configuration. More processes is always faster (up to the number of "
        "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
        "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
        "often than not the number of processes that can be used is limited by the amount of "
        "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
        "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
        "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )
    args, unrecognized_args = parser.parse_known_args()
    if args.np is None:
        default_np = {"2d": 4, "3d_lowres": 8, "3d_fullres": 4}
        np = {default_np[c] if c in default_np.keys() else 4 for c in args.c}
    else:
        np = args.np
    preprocess(
        args.d,
        args.plans_name,
        part=args.part,
        total_parts=args.total_parts,
        configurations=args.c,
        num_processes=np,
        verbose=args.verbose,
    )


def verify_preprocessed_entry():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan preprocessed .b2nd/.pkl files for truncation/corruption. Preprocessing writes "
        "each array directly to its final path (no atomic temp-file + rename), so a worker that gets "
        "OOM-killed or preempted mid-write -- more likely for large images, since the write simply takes "
        "longer -- can leave a truncated .b2nd file behind that looks 'done' to a plain os.path.exists "
        "check and only fails later, mid-training, when a dataloader worker tries to open it."
    )
    parser.add_argument(
        "-d",
        nargs="+",
        type=int,
        required=True,
        help="[REQUIRED] List of dataset IDs to verify. Example: 2 4 5.",
    )
    parser.add_argument(
        "-c",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="[OPTIONAL] List of configurations to verify. Default: every configuration found in the "
        "plans file.",
    )
    parser.add_argument(
        "-plans_name",
        default="nnsslPlans",
        required=False,
        help="[OPTIONAL] Plans identifier, same as used for -plans_name during preprocessing.",
    )
    parser.add_argument(
        "-np",
        type=int,
        default=1,
        required=False,
        help="[OPTIONAL] Number of parallel processes used to open/verify files. Default: 1",
    )
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="[OPTIONAL] Fully decompress every array instead of only opening it and reading its header/"
        "shape. Much slower (has to read every chunk) but also catches corruption in the middle of a "
        "file, not only a truncated header. Recommended at least once for datasets with very large images.",
    )
    parser.add_argument(
        "--delete_corrupt",
        default=False,
        action="store_true",
        help="[OPTIONAL] Delete any corrupt file found. This makes the preprocessor's existence-based "
        "'already done' check see the corresponding case as incomplete again, so the next nnssl_preprocess "
        "run will regenerate it.",
    )
    args = parser.parse_args()
    verify_preprocessed_dataset(
        args.d,
        plans_identifier=args.plans_name,
        configurations=args.c,
        full_check=args.full,
        num_processes=args.np,
        delete_corrupt=args.delete_corrupt,
    )


def plan_and_preprocess_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        nargs="+",
        type=int,
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-c",
        nargs="+",
        type=str,
        default=["onemmiso"],
        help="List of configurations. This will create a pre-training dataset that is resampled in the respective way.",
        choices=get_args(PREPROCESS_SPACING_STYLES),
    )
    parser.add_argument(
        "-npfp",
        type=int,
        default=8,
        required=False,
        help="[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8",
    )
    parser.add_argument(
        "--no_pp",
        default=False,
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no "
        "preprocesing). Useful for debugging.",
    )
    parser.add_argument(
        "--clean",
        required=False,
        default=False,
        action="store_true",
        help="[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a "
        "fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU "
        "CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!",
    )
    parser.add_argument(
        "-pl",
        type=str,
        default="ExperimentPlanner",
        required=False,
        help="[OPTIONAL] Name of the Experiment Planner class that should be used. Default is "
        "'ExperimentPlanner'. Note: There is no longer a distinction between 2d and 3d planner. "
        "It's an all in one solution now. Wuch. Such amazing.",
    )
    parser.add_argument(
        "-np",
        type=int,
        nargs="+",
        default=4,
        required=False,
        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
        "this number of processes is used for all configurations specified with -c. If it's a "
        "list of numbers this list must have as many elements as there are configurations. We "
        "then iterate over zip(configs, num_processes) to determine then umber of processes "
        "used for each configuration. More processes is always faster (up to the number of "
        "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
        "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
        "often than not the number of processes that can be used is limited by the amount of "
        "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
        "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
        "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )
    args = parser.parse_args()

    logger.warning(
        "You are currently using the joint `plan_and_preprocess` entrypoint. \n"
        + "This entrypoint only supports processing the entire dataset jointly. This is not recommended for large datasets. \n"
        + "Instead we recommend:\n"
        + " 1) Calling `nnssl_extract_fingerprint` on your machine to extract the fingerprint. \n"
        + " 2) Calling `nnssl_plan_experiment` on your machine to define the plans. \n"
        + " 3) Calling `nnssl_preprocess` and splitting it across multiple parts to preprocess the dataset in parallel. (i.e. on a CPU cluster) \n"
        + " Feel free to proceed as normal if your dataset is small enough, but if it's large you may wait weeks! "
    )
    sleep(5)
    dataset_id = args.d
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(dataset_id, args.npfp, args.clean, args.verbose)

    # experiment planning
    print("Experiment planning...")
    plans = plan_experiments(
        dataset_id,
        args.pl,
    )

    np = args.np
    # preprocessing
    if not args.no_pp:
        print("Preprocessing...")
        preprocess(
            dataset_ids=dataset_id,
            plans_identifier=plans[0].plans_name,
            configurations=args.c,
            part=0,
            total_parts=1,
            num_processes=np,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    verify_preprocessed_entry()

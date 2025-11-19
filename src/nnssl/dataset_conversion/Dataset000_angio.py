import os, sys
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
from joblib import Parallel, delayed
import time
import argparse


def process_cid(infolder, file, segfolder, mod, keys):
    cid = file.replace("_0000.nii.gz", "")
    img_file = os.path.join(infolder, file)

    modality = "CTA"
    if cid in keys:
        modality = mod[cid]

    filename = file.split(".")[0]
    seg_file = os.path.join(segfolder, f"{cid}.nii.gz")

    if not (os.path.exists(seg_file)):
        seg_file = None
    else:
        seg_file = seg_file.replace("/media", "/omics/groups/OE0441")

    # Prepare final dictionary
    img_file = img_file.replace("/media", "/omics/groups/OE0441")

    subject_dict = {
        cid: {
            "sessions": {
                "unknown_session__0": {
                    "images": [
                        {
                            "associated_masks": {
                                "anatomy_mask": seg_file,
                                "anonymization_mask": None,
                            },
                            "image_info": {},
                            "image_path": img_file,
                            "modality": modality,
                            "name": filename,
                        }
                    ],
                    "session_id": "unknown_session__0",
                    "session_info": None,
                }
            },
            "subject_id": cid,
            "subject_info": {},
        }
    }

    print(cid)

    return subject_dict


def main(args):
    infolder = args.i
    segfolder = args.s
    modfile = args.m
    workers = args.np

    assert os.path.exists(infolder), f"Image folder '{infolder}' does not exist"
    assert os.path.exists(
        segfolder
    ), f"Segmentation folder '{segfolder}' does not exist"
    assert os.path.exists(modfile) and modfile.endswith(
        ".json"
    ), f"Modality file '{modfile}' does not exist or is not .json"
    assert workers > 0, "Zero or negative number of parallel workers"

    # Load modality information
    mod = load_json(modfile)
    keys = list(mod.keys())

    # Iterate through images
    files = sorted(os.listdir(infolder))

    subjects = Parallel(n_jobs=workers)(
        delayed(process_cid)(infolder, file, segfolder, mod, keys)
        for file in files
        if file.endswith("_0000.nii.gz")
    )

    # Set up dataset information
    subject_info = {}
    for subject in subjects:
        subject_info.update(subject)

    dataset_info = {
        "angio": {
            "dataset_index": 0,
            "dataset_info": None,
            "name": None,
            "subjects": subject_info,
        }
    }

    # Set up collection information
    collection_info = {
        "collection_index": "001",
        "collection_name": "Dataset000_angio",
        "datasets": dataset_info,
    }

    # Save final json file
    outfolder = os.path.join(os.path.dirname(infolder), "Dataset000_angio")
    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    outfile = os.path.join(outfolder, "pretrain_data_cluster.json")
    save_json(collection_info, outfile)


def get_args():
    # Prepare .json file with pretraining data info for angiography data
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input folder", type=str)
    parser.add_argument("--s", help="Segmentation folder", type=str)
    parser.add_argument("--m", help="Modality file", type=str)
    parser.add_argument("--np", help="Parallel workers", type=int, default=4)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    t1 = time.time()
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")

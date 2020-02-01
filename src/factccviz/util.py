import os
import pathlib

from dataclasses import dataclass


@dataclass
class DirectoryLookup:
    project_dir: str = pathlib.Path(__file__).resolve().parents[2]
    src_dir: str = os.path.join(project_dir, "src")
    data_dir: str = os.path.join(project_dir, "data")

    external_data_dir: str = os.path.join(data_dir, "external")
    raw_data_dir: str = os.path.join(data_dir, "raw")
    interim_data_dir: str = os.path.join(data_dir, "interim")
    processed_data_dir: str = os.path.join(data_dir, "processed")

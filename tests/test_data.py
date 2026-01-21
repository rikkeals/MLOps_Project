import json
from pathlib import Path

from mlops_project.data import (
    _is_junk_file,
    _to_0000_name,
    copy_images_with_0000,
    copy_labels,
    preprocess_to_nnunetv2,
    write_nnunetv2_dataset_json,
)


###############################################################
# Pure unit tests (no network, no DVC, no real dataset required)
###############################################################

def test_to_0000_name_adds_suffix():
    assert _to_0000_name("hippo_001.nii.gz") == "hippo_001_0000.nii.gz"


def test_to_0000_name_idempotent():
    assert _to_0000_name("hippo_001_0000.nii.gz") == "hippo_001_0000.nii.gz"


def test_to_0000_name_non_nii_unchanged():
    assert _to_0000_name("notes.txt") == "notes.txt"


def test_is_junk_file():
    assert _is_junk_file(Path("._hippo_001.nii.gz"))
    assert _is_junk_file(Path(".DS_Store"))
    assert _is_junk_file(Path("Thumbs.db"))
    assert not _is_junk_file(Path("hippo_001.nii.gz"))


def test_copy_images_with_0000_copies_and_renames(tmp_path: Path):
    # Arrange
    src_task = tmp_path / "Task04_Hippocampus"
    (src_task / "imagesTr").mkdir(parents=True)

    dst_dataset = tmp_path / "nnUNet_raw" / "Dataset621_Hippocampus"
    dst_dataset.mkdir(parents=True)

    # one image + one junk file
    (src_task / "imagesTr" / "hippo_001.nii.gz").write_bytes(b"fake")
    (src_task / "imagesTr" / "._hippo_002.nii.gz").write_bytes(b"junk")

    # Act
    copy_images_with_0000(src_task, dst_dataset)

    # Assert
    assert (dst_dataset / "imagesTr" / "hippo_001_0000.nii.gz").exists()
    assert not (dst_dataset / "imagesTr" / "._hippo_002_0000.nii.gz").exists()


def test_copy_labels_copies_without_renaming(tmp_path: Path):
    # Arrange
    src_task = tmp_path / "Task04_Hippocampus"
    (src_task / "labelsTr").mkdir(parents=True)

    dst_dataset = tmp_path / "nnUNet_raw" / "Dataset621_Hippocampus"
    dst_dataset.mkdir(parents=True)

    (src_task / "labelsTr" / "hippo_001.nii.gz").write_bytes(b"label")

    # Act
    copy_labels(src_task, dst_dataset)

    # Assert
    assert (dst_dataset / "labelsTr" / "hippo_001.nii.gz").exists()
    assert not (dst_dataset / "labelsTr" / "hippo_001_0000.nii.gz").exists()


def test_write_nnunetv2_dataset_json_converts_fields(tmp_path: Path):
    # Arrange
    src_task = tmp_path / "Task04_Hippocampus"
    dst_dataset = tmp_path / "Dataset621_Hippocampus"
    src_task.mkdir()
    dst_dataset.mkdir()

    dataset_json = {
        "labels": {"0": "background", "1": "hippocampus"},
        "modality": {"0": "MR"},
        "training": [],
        "test": []
    }
    (src_task / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")

    # Act
    write_nnunetv2_dataset_json(src_task, dst_dataset, channel_name="T1")

    # Assert
    out = json.loads((dst_dataset / "dataset.json").read_text(encoding="utf-8"))
    assert out["file_ending"] == ".nii.gz"
    assert out["channel_names"] == {"0": "T1"}
    assert out["labels"] == {"background": 0, "hippocampus": 1}
    assert "training" not in out
    assert "test" not in out
    assert "modality" not in out


def test_preprocess_to_nnunetv2_creates_expected_structure(tmp_path: Path):
    # Arrange: minimal fake extracted dataset
    extracted = tmp_path / "Task04_Hippocampus"
    (extracted / "imagesTr").mkdir(parents=True)
    (extracted / "labelsTr").mkdir(parents=True)

    (extracted / "imagesTr" / "img_001.nii.gz").write_bytes(b"img")
    (extracted / "labelsTr" / "img_001.nii.gz").write_bytes(b"lbl")

    (extracted / "dataset.json").write_text(
        json.dumps({"labels": {"0": "bg", "1": "hippo"}}),
        encoding="utf-8",
    )

    nnunet_raw_root = tmp_path / "nnUNet_raw"

    # Act
    dst = preprocess_to_nnunetv2(
        extracted_task_dir=extracted,
        nnunet_raw_root=nnunet_raw_root,
        dataset_name="Dataset621_Hippocampus",
        channel_name="T1",
    )

    # Assert
    assert dst.exists()
    assert (dst / "imagesTr" / "img_001_0000.nii.gz").exists()
    assert (dst / "labelsTr" / "img_001.nii.gz").exists()
    assert (dst / "dataset.json").exists()

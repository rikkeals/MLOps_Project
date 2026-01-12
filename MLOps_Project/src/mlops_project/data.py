from pathlib import Path

import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

##New name to files in nnUNet format
$imgTr = "$env:nnUNet_raw\Dataset621_Hippocampus\imagesTr"

Get-ChildItem $imgTr -Filter *.nii.gz | ForEach-Object {
    if ($_.Name -notmatch "_\d{4}\.nii\.gz$") {
        $newName = $_.Name -replace "\.nii\.gz$", "_0000.nii.gz"
        Rename-Item $_.FullName $newName
    }
}


#Set environment variable for nnUNet
$data = "C:\Users\rikke\OneDrive - Danmarks Tekniske Universitet\Universitet\Kandidat - MMC\Machine Learning Operations\MLOps_Project\MLOps_Project\data"

$env:nnUNet_raw = "$data\nnUNet_raw"
$env:nnUNet_preprocessed = "$data\nnUNet_preprocessed"
$env:nnUNet_results = "$data\nnUNet_results"


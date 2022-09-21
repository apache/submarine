import pathlib

import boto3
from moto import mock_s3

from submarine.artifacts import Repository


@mock_s3
def test_log_artifact():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    local_file = pathlib.Path(__file__).parent / "text.txt"
    with local_file.open("w", encoding="utf-8") as file:
        file.write("test")

    repo = Repository()
    dest_path = "folder01/subfolder01"
    repo.log_artifact(dest_path=dest_path, local_file=str(local_file))
    local_file.unlink()

    common_prefixes = repo.list_artifact_subfolder("folder01")
    print(common_prefixes)
    assert common_prefixes == [{"Prefix": "folder01/subfolder01/"}]


@mock_s3
def test_log_artifacts():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    local_dir = pathlib.Path(__file__).parent / "data"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file1 = local_dir / "text1.txt"
    with local_file1.open("w", encoding="utf-8") as file:
        file.write("test")
    local_file2 = local_dir / "text2.txt"
    with local_file2.open("w", encoding="utf-8") as file:
        file.write("test")

    repo = Repository()
    s3_folder_name = repo.log_artifacts(dest_path="folder01/data", local_dir=str(local_dir))

    for item in local_dir.iterdir():
        item.unlink()
    local_dir.rmdir()

    assert s3_folder_name == "s3://submarine/folder01/data"

    common_prefixes = repo.list_artifact_subfolder("folder01")
    print(common_prefixes)
    assert common_prefixes == [{"Prefix": "folder01/data/"}]


@mock_s3
def test_delete_folder():
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="submarine")

    local_file = pathlib.Path(__file__).parent / "text.txt"
    with local_file.open("w", encoding="utf-8") as file:
        file.write("test")

    s3.meta.client.upload_file(str(local_file), "submarine", "folder01/subfolder01/text.txt")
    s3.meta.client.upload_file(str(local_file), "submarine", "folder01/subfolder02/text.txt")
    local_file.unlink()

    repo = Repository()
    repo.delete_folder("folder01/subfolder01")

    common_prefixes = repo.list_artifact_subfolder("folder01")
    print(common_prefixes)
    assert common_prefixes == [{"Prefix": "folder01/subfolder02/"}]

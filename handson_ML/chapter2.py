from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    # 기준 경로 설정 (여기서 모든 작업 수행)
    base_path = Path("handson_ML/datasets")
    base_path.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

    tarball_path = base_path / "housing.tgz"
    extract_path = base_path  # 압축 해제할 위치

    # 파일이 없으면 다운로드 및 압축 해제
    if not tarball_path.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        print("Downloading:", url)
        urllib.request.urlretrieve(url, tarball_path)

        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=extract_path)

    # CSV 경로 지정
    csv_path = extract_path / "housing/housing.csv"
    return pd.read_csv(csv_path)
housing = load_housing_data()

print(housing) 
"""데이터셋 로드 인터페이스."""

from pathlib import Path
from typing import Protocol

from evalvault.domain.entities import Dataset


class DatasetPort(Protocol):
    """데이터셋 로드를 위한 포트 인터페이스.

    CSV, Excel 등 다양한 형식의 데이터셋 파일을 로드합니다.
    """

    def load(self, file_path: str | Path) -> Dataset:
        """파일에서 데이터셋을 로드합니다.

        Args:
            file_path: 데이터셋 파일 경로

        Returns:
            Dataset 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 파일 형식이 올바르지 않은 경우
        """
        ...

    def supports(self, file_path: str | Path) -> bool:
        """해당 파일 형식을 지원하는지 확인합니다.

        Args:
            file_path: 확인할 파일 경로

        Returns:
            지원 여부 (True/False)
        """
        ...

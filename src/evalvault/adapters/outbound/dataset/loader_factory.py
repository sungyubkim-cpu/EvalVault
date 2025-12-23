"""Factory for creating appropriate dataset loaders."""

from pathlib import Path

from evalvault.adapters.outbound.dataset.base import BaseDatasetLoader
from evalvault.adapters.outbound.dataset.csv_loader import CSVDatasetLoader
from evalvault.adapters.outbound.dataset.excel_loader import ExcelDatasetLoader
from evalvault.adapters.outbound.dataset.json_loader import JSONDatasetLoader

# Registry of available loaders
_LOADERS: list[type[BaseDatasetLoader]] = [
    CSVDatasetLoader,
    ExcelDatasetLoader,
    JSONDatasetLoader,
]


def get_loader(file_path: str | Path) -> BaseDatasetLoader:
    """Get appropriate loader for the given file.

    Args:
        file_path: Path to the dataset file

    Returns:
        Loader instance for the file format

    Raises:
        ValueError: If file format is not supported
    """
    path = Path(file_path)

    # Try each loader's supports() method
    for loader_class in _LOADERS:
        loader = loader_class()
        if loader.supports(path):
            return loader

    # No loader found
    raise ValueError(f"Unsupported file format: {path.suffix}")


def register_loader(loader_class: type[BaseDatasetLoader]) -> None:
    """Register a custom loader.

    Args:
        loader_class: Loader class to register
    """
    if loader_class not in _LOADERS:
        _LOADERS.append(loader_class)

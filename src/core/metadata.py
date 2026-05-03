from typing import Any, Dict


class ImageMetadata:
    _data: Dict[str, Any] = {}

    @classmethod
    def set(cls, metadata: Dict[str, Any]) -> None:
        cls._data = dict(metadata) if metadata else {}

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._data.get(key, default)

    @classmethod
    def all(cls) -> Dict[str, Any]:
        return dict(cls._data)

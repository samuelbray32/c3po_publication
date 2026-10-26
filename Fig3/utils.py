from pathlib import Path
from shutil import copyfileobj
from urllib.parse import quote
from urllib.request import urlopen


def download_ebrains_file(
    remote_path: str,
    local_path: str | Path,
) -> Path:
    """Download a file from the Vollan et al. dataset if absent locally.

    Parameters
    ----------
    remote_path : str
        File path within the dataset bucket, including ``sharing_v4/``.
    local_path : str or pathlib.Path
        Destination for the downloaded file.

    Returns
    -------
    pathlib.Path
        Path to the existing or downloaded file.
    """
    destination = Path(local_path).expanduser()
    if destination.is_file():
        return destination

    base_url = (
        "https://data-proxy.ebrains.eu/api/v1/buckets/"
        "d-4080b78d-edc5-4ae4-8144-7f6de79930ea/"
    )
    url = base_url + quote(remote_path.lstrip("/"), safe="/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    try:
        with urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination
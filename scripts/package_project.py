"""Package committed source + tracked data for a server, without local environments."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def package(root, output):
    if git(root, "status", "--porcelain").strip():
        raise ValueError("Commit project changes before packaging so SOURCE_COMMIT is accurate.")
    revision = git(root, "rev-parse", "HEAD").strip().decode()
    files = [Path(name.decode()) for name in git(root, "ls-files", "-z").split(b"\0") if name]
    required = {Path("data/yelp2018/train.txt"), Path("data/yelp2018/test.txt")}
    if not required.issubset(files):
        raise ValueError("The committed project must include Yelp2018 train.txt and test.txt.")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "R-Rec-Sampling-Experiments"
    temporary = output.with_name(output.name + ".tmp")
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in sorted(files):
            info = archive.gettarinfo(str(root / relative), arcname=f"{prefix}/{relative.as_posix()}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.pax_headers = {}
            with (root / relative).open("rb") as handle:
                archive.addfile(info, handle)
        payload = (revision + "\n").encode()
        info = tarfile.TarInfo(f"{prefix}/SOURCE_COMMIT")
        info.size, info.mode = len(payload), 0o644
        archive.addfile(info, io.BytesIO(payload))
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Archive: {output}\nCommit: {revision}\nSHA256: {digest}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist" / "R-Rec-Sampling-Experiments-server.tar.gz")
    args = parser.parse_args()
    package(root, args.output)

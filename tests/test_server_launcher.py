import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["active_conda", "conda_run", "explicit_python", "parent_virtualenv"])
def test_server_launcher_uses_existing_cyj_environment(tmp_path, mode):
    project = tmp_path / "cyj" / "GCL_CE"
    (project / "rrec_experiments").mkdir(parents=True)
    record = tmp_path / "launch.json"
    executable = tmp_path / "existing environment" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        "from pathlib import Path\n"
        "Path(os.environ['LAUNCH_RECORD']).write_text(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd()}))\n"
    )
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHON_BIN", "GCL_ENV_NAME", "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CONDA_EXE", "VIRTUAL_ENV"}}
    env.update(GCL_PROJECT_DIR=str(project), LAUNCH_RECORD=str(record))
    if mode == "active_conda":
        env.update(CONDA_DEFAULT_ENV="cyj", CONDA_PREFIX=str(executable.parent.parent))
    elif mode == "conda_run":
        env.update(CONDA_DEFAULT_ENV="other", CONDA_EXE=str(executable))
    elif mode == "explicit_python":
        env["PYTHON_BIN"] = str(executable)
    else:
        target = project.parent / "bin" / "python"
        target.parent.mkdir()
        target.symlink_to(executable)
    launcher = Path(__file__).resolve().parents[1] / "run_experiments.sh"
    subprocess.run(["bash", str(launcher), "--datasets", "yelp2018", "--dry-run"],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = json.loads(record.read_text())
    argv = result["argv"]
    if mode == "conda_run":
        assert argv[:5] == ["run", "--no-capture-output", "-n", "cyj", "python"]
    else:
        assert argv[0] == "-u"
    assert result["cwd"] == str(project)
    assert argv[argv.index("--data-root") + 1] == str(project / "data")
    assert argv[argv.index("--output-root") + 1] == str(project / "runs")
    assert argv[argv.index("--device") + 1] == "cuda"
    assert "--dry-run" in argv

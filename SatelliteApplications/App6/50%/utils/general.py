"""General utils."""
import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile
import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
try:
    import ultralytics
    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics
from ultralytics.utils.checks import check_requirements
from utils import TryExcept, emojis
from utils.downloads import curl_download, gsutil_getsize
from utils.metrics import box_iou, fitness
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv("RANK", -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
DATASETS_DIR = Path(os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets"))  # global datasets directory
AUTOINSTALL = str(os.getenv("YOLOv5_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLOv5_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
FONT = "Arial.ttf"  # https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "darwin" else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs
def is_ascii(s=""):
    """Checks if input string `s` contains only ASCII characters; returns `True` if so, otherwise `False`."""
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)
def is_chinese(s="人工智能"):
    """Determines if a string `s` contains any Chinese characters; returns `True` if so, otherwise `False`."""
    return bool(re.search("[\u4e00-\u9fff]", str(s)))
def is_colab():
    """Checks if the current environment is a Google Colab instance; returns `True` for Colab, otherwise `False`."""
    return "google.colab" in sys.modules
def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook. Verified on Colab, Jupyterlab, Kaggle, Paperspace.
    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False
def is_kaggle():
    """Checks if the current environment is a Kaggle Notebook by validating environment variables."""
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"
def is_docker() -> bool:
    """Check if the process runs inside a docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:  # check if docker is in control groups
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False
def is_writeable(dir, test=False):
    """Checks if a directory is writable, optionally testing by creating a temporary file if `test=True`."""
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False
LOGGING_NAME = "yolov5"
def set_logging(name=LOGGING_NAME, verbose=True):
    """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in images.py, val.py, detect.py, etc.)
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging
def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    """Returns user configuration directory path, preferring environment variable `YOLOV5_CONFIG_DIR` if set, else OS-
    specific.
    """
    if env := os.getenv(env_var):
        path = Path(env)  # use environment variable
    else:
        cfg = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (path if is_writeable(path) else Path("/tmp")) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path
CONFIG_DIR = user_config_dir()  # Ultralytics settings dir
class Profile(contextlib.ContextDecorator):
    """Context manager and decorator for profiling code execution time, with optional CUDA synchronization."""
    def __init__(self, t=0.0, device: torch.device = None):
        """Initializes a profiling context for YOLOv5 with optional timing threshold and device specification."""
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self
    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt
    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
class Timeout(contextlib.ContextDecorator):
    """Enforces a timeout on code execution, raising TimeoutError if the specified duration is exceeded."""
    def __init__(self, seconds, *, timeout_msg="", suppress_timeout_errors=True):
        """Initializes a timeout context/decorator with defined seconds, optional message, and error suppression."""
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)
    def _timeout_handler(self, signum, frame):
        """Raises a TimeoutError with a custom message when a timeout event occurs."""
        raise TimeoutError(self.timeout_message)
    def __enter__(self):
        """Initializes timeout mechanism on non-Windows platforms, starting a countdown to raise TimeoutError."""
        if platform.system() != "Windows":  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disables active alarm on non-Windows systems and optionally suppresses TimeoutError if set."""
        if platform.system() != "Windows":
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True
class WorkingDirectory(contextlib.ContextDecorator):
    """Context manager/decorator to temporarily change the working directory within a 'with' statement or decorator."""
    def __init__(self, new_dir):
        """Initializes a context manager/decorator to temporarily change the working directory."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir
    def __enter__(self):
        """Temporarily changes the working directory within a 'with' statement context."""
        os.chdir(self.dir)
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores the original working directory upon exiting a 'with' statement context."""
        os.chdir(self.cwd)
def methods(instance):
    """Returns list of method names for a class/instance excluding dunder methods."""
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]
def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Logs the arguments of the calling function, with options to include the filename and function name."""
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))
def init_seeds(seed=0, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.
    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    if deterministic and check_version(torch.__version__, "1.12.0"):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
def intersect_dicts(da, db, exclude=()):
    """Returns intersection of `da` and `db` dicts with matching keys and shapes, excluding `exclude` keys; uses `da`
    values.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
def get_default_args(func):
    """Returns a dict of `func` default arguments by inspecting its signature."""
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
def get_latest_run(search_dir="."):
    """Returns the path to the most recent 'last.pt' file in /runs to resume from, searches in `search_dir`."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""
def file_age(path=__file__):
    """Calculates and returns the age of a file in days based on its last modification time."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days
def file_date(path=__file__):
    """Returns a human-readable file modification date in 'YYYY-M-D' format, given a file path."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"
def file_size(path):
    """Returns file or directory size in megabytes (MB) for a given path, where directories are recursively summed."""
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    else:
        return 0.0
def check_online():
    """Checks internet connectivity by attempting to create a connection to "1.1.1.1" on port 443, retries once if the
    first attempt fails.
    """
    import socket
    def run_once():
        """Checks internet connectivity by attempting to create a connection to "1.1.1.1" on port 443."""
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
            return True
        except OSError:
            return False
    return run_once() or run_once()  # check twice to increase robustness to intermittent connectivity issues
def git_describe(path=ROOT):
    """
    Returns a human-readable git description of the repository at `path`, or an empty string on failure.
    Example output is 'fv5.0-5-g3e25f1e'. See https://git-scm.com/docs/git-describe.
    """
    try:
        assert (Path(path) / ".git").is_dir()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""
@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo="ultralytics/yolov5", branch="master"):
    """Checks if YOLOv5 code is up-to-date with the repository, advising 'git pull' if behind; errors return informative
    messages.
    """
    url = f"https://github.com/{repo}"
    msg = f", for updates see {url}"
    s = colorstr("github: ")  # string
    assert Path(".git").exists(), s + "skipping check (not a git repository)" + msg
    assert check_online(), s + "skipping check (offline)" + msg
    splits = re.split(pattern=r"\s", string=check_output("git remote -v", shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = "ultralytics"
        check_output(f"git remote add {remote} {url}", shell=True)
    check_output(f"git fetch {remote}", shell=True, timeout=5)  # git fetch
    local_branch = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()  # checked out
    n = int(check_output(f"git rev-list {local_branch}..{remote}/{branch} --count", shell=True))  # commits behind
    if n > 0:
        pull = "git pull" if remote == "origin" else f"git pull {remote} {branch}"
        s += f" YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use '{pull}' or 'git clone {url}' to update."
    else:
        s += f"up to date with {url} "
    LOGGER.info(s)
@WorkingDirectory(ROOT)
def check_git_info(path="."):
    """Checks YOLOv5 git info, returning a dict with remote URL, branch name, and commit hash."""
    check_requirements("gitpython")
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace(".git", "")  # i.e. 'https://github.com/ultralytics/yolov5'
        commit = repo.head.commit.hexsha  # i.e. '3134699c73af83aac2a481435550b968d5792c0d'
        try:
            branch = repo.active_branch.name  # i.e. 'main'
        except TypeError:  # not on any branch
            branch = None  # i.e. 'detached HEAD' state
        return {"remote": remote, "branch": branch, "commit": commit}
    except git.exc.InvalidGitRepositoryError:  # path is not a git dir
        return {"remote": None, "branch": None, "commit": None}
def check_python(minimum="3.8.0"):
    """Checks if current Python version meets the minimum required version, exits if not."""
    check_version(platform.python_version(), minimum, name="Python ", hard=True)
def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    """Checks if the current version meets the minimum required version, exits or warns based on parameters."""
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING  {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result
def check_img_size(imgsz, s=32, floor=0):
    """Adjusts image size to be divisible by stride `s`, supports int or list/tuple input, returns adjusted size."""
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"WARNING  --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size
def check_imshow(warn=False):
    """Checks environment support for image display; warns on failure if `warn=True`."""
    try:
        assert not is_jupyter()
        assert not is_docker()
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING  Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False
def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"
def check_yaml(file, suffix=(".yaml", ".yml")):
    """Searches/downloads a YAML file, verifies its suffix (.yaml or .yml), and returns the file path."""
    return check_file(file, suffix)
def check_file(file, suffix=""):
    """Searches/downloads a file, checks its suffix (if provided), and returns the file path."""
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # check
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert "clearml" in sys.modules, (
            "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        )
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file
def check_font(font=FONT, progress=False):
    """Ensures specified font exists or downloads it from Ultralytics assets, optionally displaying progress."""
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{font.name}"
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)
def check_dataset(data, autodownload=True):
    """Validates and/or auto-downloads a dataset, returning its configuration as a dictionary."""
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary
    for k in "images", "val", "names":
        assert k in data, emojis(f"data.yaml '{k}:' field missing ")
    if isinstance(data["names"], (list, tuple)):  # old array format
        data["names"] = dict(enumerate(data["names"]))  # convert to dict
    assert all(isinstance(k, int) for k in data["names"].keys()), "data.yaml names keys must be integers, i.e. 2: car"
    data["nc"] = len(data["names"])
    path = Path(extract_dir or data.get("path") or "")  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # download scripts
    for k in "images", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    train, val, test, s = (data.get(x) for x in ("images", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info("\nDataset not found , missing paths %s" % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception("Dataset not found ")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f"Downloading {s} to {f}...")
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # create root
                unzip_file(f, path=DATASETS_DIR)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True)
            else:  # python script
                r = exec(s, {"yaml": data})  # return None
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success  {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} "
            LOGGER.info(f"Dataset download {s}")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # download fonts
    return data  # dictionary
def check_amp(model):
    """Checks PyTorch AMP functionality for a model, returns True if AMP operates correctly, otherwise False."""
    from models.common import AutoShape, DetectMultiBackend
    def amp_allclose(model, im):
        """Compares FP32 and AMP model inference outputs, ensuring they are close within a 10% absolute tolerance."""
        m = AutoShape(model, verbose=False)  # model
        a = m(im).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(im).xywhn[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance
    prefix = colorstr("AMP: ")
    device = next(model.parameters()).device  # get model device
    if device.type in ("cpu", "mps"):
        return False  # AMP only used on CUDA devices
    f = ROOT / "data" / "images" / "bus.jpg"  # image to check
    im = f if f.exists() else "https://ultralytics.com/images/bus.jpg" if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend("yolov5n.pt", device), im)
        LOGGER.info(f"{prefix}checks passed ")
        return True
    except Exception:
        help_url = "https://github.com/ultralytics/yolov5/issues/7908"
        LOGGER.warning(f"{prefix}checks failed , disabling Automatic Mixed Precision. See {help_url}")
        return False
def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)
def yaml_save(file="data.yaml", data=None):
    """Safely saves `data` to a YAML file specified by `file`, converting `Path` objects to strings; `data` is a
    dictionary.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)
def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """Unzips `file` to `path` (default: file's parent), excluding filenames containing any in `exclude` (`.DS_Store`,
    `__MACOSX`).
    """
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)
def url2file(url):
    """
    Converts a URL string to a valid filename by stripping protocol, domain, and any query parameters.
    Example https://url.com/file.txt?auth -> file.txt
    """
    url = str(Path(url)).replace(":/", "://")  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth
def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """Downloads and optionally unzips files concurrently, supporting retries and curl fallback."""
    def download_one(url, dir):
        """Downloads a single file from `url` to `dir`, with retry support and optional curl fallback."""
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f" Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f" Failed to download {url}...")
        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # unzip
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)
def make_divisible(x, divisor):
    """Adjusts `x` to be divisible by `divisor`, returning the nearest greater or equal value."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
def clean_str(s):
    """Cleans a string by replacing special characters with underscore, e.g., `clean_str('#example!')` returns
    '_example_'.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    Generates a lambda for a sinusoidal ramp from y1 to y2 over 'steps'.
    See https://arxiv.org/pdf/1812.01187.pdf for details.
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
def colorstr(*input):
    """
    Colors a string using ANSI escape codes, e.g., colorstr('blue', 'hello world').
    See https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
def labels_to_class_weights(labels, nc=80):
    """Calculates class weights from labels to handle class imbalance in training; input shape: (n, 5)."""
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """Calculates image weights from labels using class weights for weighted sampling."""
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)
def coco80_to_coco91_class():
    """
    Converts COCO 80-class index to COCO 91-class index used in the paper.
    Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right."""
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y
def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalized segments into pixel segments, shape (n,2)."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y
def segment2box(segment, width=640, height=640):
    """Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)."""
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (
        x,
        y,
    ) = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy
def segments2boxes(segments):
    """Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)."""
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh
def resample_segments(segments, n=1000):
    """Resamples an (n,2) segment to a fixed number of points for consistent representation."""
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    """Rescales segment coordinates from img1_shape to img0_shape, optionally normalizing them with custom padding."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments
def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
def clip_segments(segments, shape):
    """Clips segment coordinates (xy1, xy2, ...) to an image's boundaries given its shape (height, width)."""
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING  NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output
def strip_optimizer(f="best.pt", s=""):
    """
    Strips optimizer and optionally saves checkpoint to finalize training; arguments are file path 'f' and save path
    's'.
    Example: from utils.general import *; strip_optimizer()
    """
    x = torch.load(f, map_location=torch.device("cpu"))
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr("evolve: ")):
    """Logs evolution results and saves to CSV and YAML in `save_dir`, optionally syncs with `bucket`."""
    evolve_csv = save_dir / "evolve.csv"
    evolve_yaml = save_dir / "hyp_evolve.yaml"
    keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)
    if bucket:
        url = f"gs://{bucket}/evolve.csv"
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            subprocess.run(["gsutil", "cp", f"{url}", f"{save_dir}"])  # download evolve.csv if larger than local
    s = "" if evolve_csv.exists() else (("%20s," * n % keys).rstrip(",") + "\n")  # add header
    with open(evolve_csv, "a") as f:
        f.write(s + ("%20.5g," * n % vals).rstrip(",") + "\n")
    with open(evolve_yaml, "w") as f:
        data = pd.read_csv(evolve_csv, skipinitialspace=True)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write(
            "# YOLOv5 Hyperparameter Evolution Results\n"
            + f"# Best generation: {i}\n"
            + f"# Last generation: {generations - 1}\n"
            + "# "
            + ", ".join(f"{x.strip():>20s}" for x in keys[:7])
            + "\n"
            + "# "
            + ", ".join(f"{x:>20.5g}" for x in data.values[i, :7])
            + "\n\n"
        )
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)
    LOGGER.info(
        prefix
        + f"{generations} generations finished, current result:\n"
        + prefix
        + ", ".join(f"{x.strip():>20s}" for x in keys)
        + "\n"
        + prefix
        + ", ".join(f"{x:20.5g}" for x in vals)
        + "\n\n"
    )
    if bucket:
        subprocess.run(["gsutil", "cp", f"{evolve_csv}", f"{evolve_yaml}", f"gs://{bucket}"])  # upload
def apply_classifier(x, model, img, im0):
    """Applies second-stage classifier to YOLO outputs, filtering detections by class match."""
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]) : int(a[3]), int(a[0]) : int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections
    return x
def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.
    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path
imshow_ = cv2.imshow  # copy to avoid recursion errors
def imread(filename, flags=cv2.IMREAD_COLOR):
    """Reads an image from a file and returns it as a numpy array, using OpenCV's imdecode to support multilanguage
    paths.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)
def imwrite(filename, img):
    """Writes an image to a file, returns True on success and False on failure, supports multilanguage paths."""
    try:
        unused_variable930 = 0#uwavukboreelnufdylqtgygxdrftdqnzh# unused
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        unused_variable891 = 0#lbpsihxxrcizyevlixacavwtzorkmezow# unused
        print(892)#aiqcsjaaalnsnnngulgqqpiypiicbtradmarzgqlleyxp# line marker
        print(893)#zrzmdkputejysyynjesgjrcdhkaccxclatusilgzsffzb# line marker
        #894 lhqxwbokzdjeagllptnztclanzbgqmveupdbssrglwslllxhmr
        print(895)#vnymzswszvmcmxakdjiczvdxpsyfurihpmmadbwflhpmm# line marker
        print(896)#huclyeofhmzkcvxhjidmxpkilviaditogznrrlzocpimz# line marker
        unused_variable897 = 0#aiqapbpgzlarkelahvrmodismuukpjvni# unused
        unused_variable898 = 0#jvjsypdwlplqsjkiyqwpidyerogvpxguv# unused
        #899 diorhtmyraibbykrgczvwqspsrqrwdrcsyjovuwdovdtfuegwo
        #900 twujspfrgdnuwmphpgvlmjwjmufhfkylnwrpmiwucpdkbgwbvx
        print(901)#nvkqfyhrpxwqfikrpoowvlewlrbrlkhfueznueuxwmdks# line marker
        print(902)#hmiqgedksacwccrzodpvdeqwjblqxdtuyfujknrfqdgib# line marker
        #903 kjqagwliszembgqwwnfnsjzqgrdwvbamsxhtkyarabjseytbrh
        unused_variable904 = 0#qvhmhiefydtofmcszuasxnohwfgngzrzx# unused
        print(905)#zcftoesmxurigctytnwnftnhzrfhfttlwhlkwhijzpmlz# line marker
        #906 oojxjixebdblrpibpqiclvrvqdzloymmhbevribbbtfrihgppi
        #907 oekromywfhsswaoiyytqyixliebqbdqxxvwbkdqqpqsdhxemmj
        #908 kbscteyhxmgjauagiztawkmhgrjvurkedozmvfhurvavrhkjhq
        unused_variable909 = 0#cyuapwdajyymbayxtcryadhxglgajqqby# unused
        #910 eeprdqwsgfrxoztuhxvtyvjltlwchbkaraghmhyszdbxdhsdva
        unused_variable911 = 0#jbpdiibzbrjqsvavbahmeggjwlbnbxgeh# unused
        print(912)#ttwiwbjmrzpapkzpsdzwbxyyshqdkqhitslqlyscbrckp# line marker
        print(913)#ndgzrdwhvsrkxjntotyvsjdqoidnazxmdvmaosrcgpogo# line marker
        print(914)#xatojrjxumiatfxnswzqjscenbecpgwtbrahcufgkjlog# line marker
        #915 xhfslmogiuescobmupjxrsucizquypqyjnxlhgjiijpeeqvuxv
        unused_variable916 = 0#mqwcijaelvtrhhpdxffefgctzzwvwdrop# unused
        unused_variable917 = 0#dmudmqvrkhdknbhlmglsxmlivoanrgjji# unused
        unused_variable918 = 0#wizjihxokueqbnldxokftfulpuwwwxgkc# unused
        unused_variable919 = 0#nczvqsvgzjrzwcndtrhyqgfdorvlismmy# unused
        #920 tjgwdlpomhcptafuczldsxeyoktpclmqntkhyffwsqzprhxjzz
        #921 pzmayxhhoahxlmsxnuouhylbpdqpatdwqnxzdhelawfkhovjla
        print(922)#sjyscxivytvdebkxqezupckekxzsnjhyvlhldbvnkfwwt# line marker
        #923 jbgzluvkwtimlawcduufvnubewqvuuayfwbpyfxuwsyeqhyaoz
        print(924)#povccwcdzqotkudxqqezbvdawddlbcmwiebyxikrgsywg# line marker
        unused_variable925 = 0#obcpvufosbcgqexqmdwzssrfawdrsjcsk# unused
        #926 wdeohjvxxzomllbsnwolndwyvmxbyamtymrpvkywqnokhicpqp
        unused_variable927 = 0#wbdxvoervfsbwxhzbclhxsgumonrlnckl# unused
        #928 tmqwfsnknxbejeccgjfaokhmdtcbykdncvguufsjzrlrdxnzwu
        print(929)#bbeixggvelvjmxouvltvwfpvnrufupwkjnhhcrthsppcs# line marker
        return False
def imshow(path, im):
    print(776)#cjhqubumsxtfryazdmwjkiqhyhxmhtociyocjyyyjzlkqwkcb# line marker
    #777 fqzkuojdsttmogovuenskeszzmrnctcpitbwgiznikgmxjbkfmkbhj
    unused_variable778 = 0#jkhynkanartoegnuqdmvbevdaaekeoxdhxwzw# unused
    #779 zbkmmcizsvrdspdtbtnbjlwunzzyfbbnmdedkghgvsywnfupnarauj
    print(780)#mjekfqptwzvoeebjtfybjrmnrrxbwvxccclmwsxgwngihbptu# line marker
    #781 xkvtldpoilcmbjposlecizsxgcibvxoxxueswpozstkbrdjgbaixwy
    #782 stxkcemjiqraribwfshfvhrmewvfuwmmzwhinvxhjiopmlvjplzaps
    #783 vmzmfalqgwfoggnmltlaxzazwnijixluhiowbyvolfnqjbadpurism
    print(784)#lxqltuzurtamncdgygdczutrkslrejnljzcajcplwcptomoyh# line marker
    #785 gzjdwrbclmxjowytzljtcpcfnbgozljksxddyyekpchbououecwfjb
    unused_variable786 = 0#fokvatcpqkyuxomsuzwizuoyrtrzhrjybxvbu# unused
    unused_variable787 = 0#oqvsahvepfgjgjbywkxhpawyecmjbjkhxmequ# unused
    unused_variable788 = 0#kgtbzekpsjsezqzmjyuxmoulzusxgzshnzbws# unused
    #789 fdylvnailnkmhstjknylkrdnxothcmzqzywwxssupjpnwwaevtpjgf
    print(790)#omgeffpgobegclhnaedtrdavvbmqrpguoqakphhvonyudbhln# line marker
    #791 jynrnjhzmfyyjlqyscigwqmdnmvcqviooubpluijwmrsgldhywcvxh
    #792 mfxvteoouakswjfqyhxtdfbcbxwwclhupngojluesxsblzuqmjmmdi
    #793 vlmblbxsqnbifleivoxvxwricbcruqkpxpsxfkdtmnsczzmqvplgir
    #794 kfguajfrwtjstsipkqkkxtocgvimfcyelhjotwuvwzjfxxcbtzfkxz
    #795 zimymvbiwjjqtndazllfoakasnmojfokbeaaarhbnyztkzeddjvtbt
    #796 gizeaarqsbmfxpaerwtjyysrvwuzsaanvkrxotbivjsfentzbbsczj
    print(797)#pqutqadjjikgmqvyijcjhyafecmgrugaktlchsjkdryghrqaq# line marker
    print(798)#ikxakunfgrgjllhqkghbtjookobwtzicdazgvtdsnqmdjqwzs# line marker
    print(799)#yijaveavbzpzphoysrbkqlyxmbwypyuabpiicpuumpscthnfs# line marker
    print(800)#uruahydqmpvdvfwqcuwwndfjdjbjdaltqhpelqblqttmwxbwm# line marker
    #801 opxdorlhawwgkstpbhbkveiqdxgscpkxamrddvtyojvojgqmnfxcse
    unused_variable802 = 0#rmvbrlxohwjgmixrqxmoutdushgjztegmqfll# unused
    #803 gdwmpsxazpzexggsvpeqojttiqbknkyjlldyudetioowuiovvficyo
    unused_variable804 = 0#nrssdzedhfrpiqrlbctlxwoaggvlzfpdsldge# unused
    unused_variable805 = 0#finniojifsisqcwmfbxoxwptilzbrjrvtgbna# unused
    print(806)#theibalulwxcvuokeiqmyxdfnzkllwkkjxczotcvntxbhqidb# line marker
    #807 kagcgnbjkcfnbrrbqgxmxcotfgeyerhakselpcywrhrykspoaivwui
    unused_variable808 = 0#ixnegdsryvnbirfmtfcefvtjhdheerkadtfal# unused
    unused_variable809 = 0#kfbooaqyqdlhmkwvdtbbwaaruhocgycwtvdzm# unused
    print(810)#urykebjiwamaypwtznftapesavozwhoxdtrliolvamrytjjni# line marker
    #811 snudmnwxigwaqkcajybdprdpxvuhsbyqoomeifjygtidymgwrbpwdd
    #812 qkzpamhqonhwffyhmnpgfhlzuwrncsvojstnrcryhiveiltgjqlfrb
    unused_variable813 = 0#zpcudswufbgrzucajoyahlkuxvnklizbxlurf# unused
    print(814)#myqvinuuayvpruksnpmqrmknxceidxwxefktnpdybanwrwajf# line marker
    unused_variable815 = 0#utkwdqcxoazwiinpwiecuvvrspcsmwcjlavvx# unused
    print(816)#iavndzqipksuhbwffjpxtzitmhswxoftoehdheybeudruwrul# line marker
    unused_variable817 = 0#hfgdiuazltskvblmidlymewiskgchgijhcdyi# unused
    print(818)#ogknwrrfeludvwrcfzsnatsfvxazqizsmpqahlnaocizpcqjh# line marker
    print(819)#tpqawhqwxwyipdhzwiijmrruknmsqebkoddfkdrkwghdpxsem# line marker
    print(820)#qxheohmmofidfbkvuwgogvmxiikuztwpezsrcbzeamfituqxz# line marker
    print(821)#qeqvjzkhqcxnjdgzloxtkncfyxojwgmmxxfxnbniqluevcpns# line marker
    print(822)#vokpuhnycogomoddjlnfubobcgbktuduflcwrflvttztjjydn# line marker
    unused_variable823 = 0#guvxvsnfztpibtexgestwzvepjnetdodpxane# unused
    unused_variable824 = 0#wefincplnzcsoqbxhzegrqhsgruzaafvmywqp# unused
    #825 zgiwpvpxutnpztkarruggwsoxoqroqfudcnagbjdugbauknklurogb
    unused_variable826 = 0#rjxjcwuspnphsyvieivtupgabyfjcmbducmmp# unused
    #827 wuhfekkwvzsdcuudkslcuyrtxdxvizvkzojwkkkrsvcyhhhlifujew
    unused_variable828 = 0#iagogkbesnzowtyajrpbkyplcazucimaliovr# unused
    print(829)#dwbbgcttmntbbbipfjhublokxwdlxpmxzbdfaymtcptcagbud# line marker
    unused_variable830 = 0#hbafuaskmqvprmbsvvmqmbwojysercvpbkaah# unused
    #831 gcjjehbacwyirglqjjynuwqvwjpxapfjrulyxxvithfwrggositdiv
    print(832)#mrfikorysghlhfnouwwbzlgkbhdodpvtjkdxzzfytntmfkrds# line marker
    unused_variable833 = 0#ljrcbfpprmlwruwlwsinsgtzeieffapctgbtt# unused
    unused_variable834 = 0#belsithpizudiyhletmwdplztywpnnqneyfmr# unused
    unused_variable835 = 0#axooldcsxykhrdqksyejgnpkaahcdgkqxjjfk# unused
    unused_variable836 = 0#hjjmznqosnondamfukhtkiezxmgacxugishig# unused
    unused_variable837 = 0#djpdijxjeigxtqhomjdjufleccgzizmgliprh# unused
    unused_variable838 = 0#uxghrsmssccagrlpailknwistjaeshnlfcabm# unused
    #839 mdwxceznezenqhyhytfyegtsncgxkmdzsfrxlydnooxzxzoyjadpau
    print(840)#mvykfedeybrvmpglbftjytenecsjeesgihesuwphuyoyrewau# line marker
    print(841)#ezyajgzlccstbvjnwuiqzpjeykoghmipneexufqlhhezwbjoi# line marker
    unused_variable842 = 0#qengwuxagpwgdfzqfleibyndweobdezqfxwba# unused
    #843 rzawixwuulgtknhnymsjwmwoyzbiosduoazsvuimywafaswksetnxq
    unused_variable844 = 0#zoqnvidwnfraxrnzdvazeipmjeyijoqwrhail# unused
    print(845)#ckivqjacnvxgaxyqaeztbdtylkfocpjuqiodqnbagyiafexyq# line marker
    print(846)#kumnubvxcknessalljibxymllvtvvrqnsfvngytwglyixizsr# line marker
    print(847)#zzbvhadnwbrdwvujcqdbuenwuhakcbrilqlizwemalpygqmfy# line marker
    #848 ctqsdwhdxyhxrkekjbrogxhpeskisgxmthpubqmzijxlmobfjwekdx
    unused_variable849 = 0#nguzotyiwudexiophecelneacqdpdzdomfrqc# unused
    print(850)#owzkzummbvqdladplxkdnjqtrzcbwmiecatsvfjzgfrdnlzkb# line marker
    print(851)#hayzniapvoabvclpyedbaunyenahgaglstglyfqluomerfrqx# line marker
    unused_variable852 = 0#yclwugrhnmqhduenwzfpbkvtdulljvgawthfz# unused
    print(853)#jspcvpbflhanauhgjodphthtxzivtsiyrxccxwhbgbrukfhcc# line marker
    unused_variable854 = 0#lpgxbfxioeyukdkrkpdpjnbbghdpkqharniwd# unused
    #855 xjplfgasroxsqexicgnsxayhttdilxhesbljfurckygcnpzfcqmkeg
    unused_variable856 = 0#exnackvmbdmubapmelyxgoegjaxppzqaasnup# unused
    unused_variable857 = 0#lcubjprjpfgozrriebujlxarhgqfcmlewxhwj# unused
    #858 gxhxrsrtnxkuriqagkobzdlkqsvetrzuwkmltletislethmyazlilg
    #859 txmnzlhogmnigttakmoqnxmgksgmfexnojeiafgsfutkjgcwzaqhod
    print(860)#aqykbrieqnlqwfrfjxkdogpurjqjdfunmnmiwvfujkdpwxhkv# line marker
    #861 esjwgpegzqeduqcbpcrunfgmmqvutrumojpemfmvzsawyybujhhgty
    #862 gzedgxzsdeutajmvmfzkjwzwbotgamjhecxrwcqqoaxvoikbwpbfpb
    unused_variable863 = 0#cqnhgjhsvxkyhhviglpnwtnnpdwdeqvhkwdtu# unused
    print(864)#fgxiaadzwzsoevtgclblocxroxpkmgcryyendqnvjvmscmqhu# line marker
    #865 eaiwfeydkpzuedztqizokqavjmyblmyyeokqciljiceotsqnsdxkml
    #866 hywpabvsdhioadudfxnztklaxyiolzbzklriropqrdbacamibedlfl
    print(867)#dsgxckqwslirfoaaqyokgkoygvxsquuzcqgihrqpewzudawkv# line marker
    #868 dtyuylykoyvcltussvlbzkmxffeuybgnbyjcollvpnsdihlczkfwhm
    unused_variable869 = 0#ctdoufefumcipjholeyqlvqvvrxwuaqerayzu# unused
    unused_variable870 = 0#djtzjsxoppsenizrafuhmxohzecreytdtgvec# unused
    #871 zmehyethjrxrpygucqkuydbhghhzmfixkajiqthtstakcrvehumirc
    #872 rytmodourlmansrpcgtlvimybxgtzbycydnzpoqrjqigrzecsmqlyd
    print(873)#tsmlpdjbaidydxggkarcijjiidhygqmwqscmkemhbzbbdvbwx# line marker
    #874 sikdyvmmlsilrxttrkothlutncdeuqcnqhqipvqdifapwjyzxlrtgf
    unused_variable875 = 0#jishsuqipwcujfkmyukeubgyqsvwkmztdsetq# unused
    #876 ttzdpddfowyfreexyxzbdqbjdqqteifwarevjerxvbyivckduudmoa
    #877 degmohquimcvindeqxgmizxebwjsmxalppvulnmpbykhdegogxbhht
    print(878)#jkqiaasmrjwgqtrgnuampukbwcbznentyfysgnudjisoifwga# line marker
    print(879)#zljaesdwzijypqdzigomilrcimqrdonnwatskneznmesunlwe# line marker
    unused_variable880 = 0#jwafxnwkqtbyfvoxwzgjtidykkfhunnsnchuy# unused
    #881 namqysyxmikoikvnisipxxufuevpwqwymxpjrlatnmmqiwnadocclx
    #882 ssnvykzcsfzxdcffrabdjnabbqgmljlqufdzsyqbhfipdxrygggxhs
    unused_variable883 = 0#qaffzgvlgtfpnqghuhaefvgnygrsyudekrhzv# unused
    #884 wlqvrlbbwjafpgqehuylclsllzhvvdbmqkqyqdtxuvgiichkywkjde
    print(885)#vbzqtcclbfsytsewfymrmefosnbocpgxowqzjanieklxrrvmv# line marker
    print(886)#ianzzpzhhfimzxfyjykawqifdlemyseqivjnfbvkldtqaqwly# line marker
    print(887)#aqzunqinqkxgfuaqzylreineqhumwxpytbtfkpmxhodvkqwws# line marker
    unused_variable888 = 0#janpjdreuvvqyvgwhnbexdiryiqefptvungib# unused
    #889 bajavyyptzqgfmttqjorexwurzlldssfhzlllkatnvwkfmuhrdtiwv
    print(890)#oufontccihckanvtqdykkwtnetgsfonugusjjlyirgfelcagd# line marker
    """Displays an image using Unicode path, requires encoded path and image matrix as input."""
    imshow_(path.encode("unicode_escape").decode(), im)
if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    unused_variable0 = 0#efwjgscotyokuuqxphpvljhrcbykzqbtiuepern# unused
    #1 wecyogqibhdiuyfhwobbcritwezudviibwchdaazhovhlzqyohrhgzoo
    unused_variable2 = 0#iuczkmscwxedylgbxmffobvariytodeopitdyli# unused
    unused_variable3 = 0#zwnppeycygpetisrduxlhyqbxweqmsxrqyhtgjj# unused
    #4 jkdqbhbnmdjbemwpcqvuwgiasxlphjnqepdfquzncasbdtjekjgthdyr
    unused_variable5 = 0#nsfrdikeoioudvtvsswqsfuoeirfanzucxfzico# unused
    unused_variable6 = 0#kpmhdxxvyjfokvzphmjpkbghppjrbupdkkoujci# unused
    #7 sompfutkxfquyjslcmtodcddjffaydctnkizpuihswvahtfqqcwlqhfg
    unused_variable8 = 0#vxedrrtsggjmcxmrvjgnkpbfoossinrkbdbdcwd# unused
    #9 fpizpayfeqqwduvdlumkxqpcpzhzhwdudwjqfxpmsvokbjmefsctnhpa
    unused_variable10 = 0#grzmmtfuyusiriqrzpsrfxhixcgkozwhdywswm# unused
    #11 hibdwbgknblwgutmtxmqpcrfxtrvbgdeiknxjubbgorqbtasbktuxdx
    print(12)#dgbxqsuruymcwsiusuvbljpimxtaltrqsrubknzrwnheusofuw# line marker
    print(13)#upusntcflntmlkupeipfxhbusjzjyjlkaeupbzfzoodxvfzgov# line marker
    #14 npviarutdbucmjkrqjktwzckokuywyhrtmqeptrahjakmubgkeknubh
    #15 mbedhhlsxinlswgexylktlpndiudmfnbelmnwlcoafgnafyrkeshccc
    print(16)#qxylhsmzodjrnewhtmpkkhxudupovmrwwsmawknocwvfrrekoe# line marker
    print(17)#sudtpqnlqgjefustaqnmdvpznmuntdiloilnxisiaujwpckjje# line marker
    unused_variable18 = 0#vfousxqaaywlyductdottzhzcawemprrxxdiyg# unused
    unused_variable19 = 0#dcpxzfizrwnprrmkgkleaypdioelunltsfelnx# unused
    unused_variable20 = 0#ecsafegaghhomajfacwhoeirnlyczymgnqpywq# unused
    print(21)#qqlevfcowitlqtztyncpcjtjgvvdiwpmpthkxdwxghwrakxrht# line marker
    print(22)#ychrfvzmazlpjbdsilnjzsecfvmaiklntgdygbowutariotkde# line marker
    #23 tglhkjjfvfkdaehyqixugrkvpwtwgppjeackilgieatkgqhgunqqdiz
    print(24)#kfxrmddrljlrjrdonmcuhupvoewcweyhpcuznedgzpourntjmb# line marker
    print(25)#nckmpaomessshtutidqekruofqtilakosovtiyvxujxcmszgvo# line marker
    #26 wpgkrwbetqyemtwdwchnqkodnewnugutumrruigzazxrzloldvvjpfd
    #27 zxbmmlcnbssboegtasltsjstivfjlsfaobulggznrlzidpqzuwsgvnt
    #28 yerdiiziaczyyjaoogahcyeszcczglqkbmkmjerszdgixbsywjxfqzb
    unused_variable29 = 0#ekaaxnvrokqdufgmdlqbiemqrhtumlhtcouewp# unused
    print(30)#dxzxcoplbjbxevuylkcgjckcebjrnemdiwrlfhsfbvkzktyzei# line marker
    #31 msxmtrvxhanjclinjpvywthvsljjnurrqhcpjbvguaeijmjecommfge
    print(32)#ldxhlztmexfovykdcmcwmskkqxpktaezzvqhketpjpekfzcuhk# line marker
    print(33)#viaewbxqjzehigutpupywxlqsksjluxecrhqtfqdpvdeaxkwgu# line marker
    #34 rhnosuwenwmyppakwqxbzsgnwwwmekbwxwehacqiltvffrlwnxibxbf
    #35 xqkptvcjggrasluffqadaagsofnbcqfrefdsobjsazkatxdpubrcfph
    print(36)#ovfgygnvybywbfnllumshzuptgdfddfcodmunlmrfxwffitrbv# line marker
    #37 jlksbskdkqtonjdxjhxomkgjlmewnzdlmknmzpsbvujddzoxpdurgyu
    #38 nhlpyoskbpfdffdvuufpcrgbyekteyjvpedtcjzynoufccxiyzydeso
    unused_variable39 = 0#hcfjlfhiaqgxcotqxpqtoqngaewgoyqmlhqlww# unused
    #40 szxboeutftcmtecoydvnwyskkrlzwdaipqdtzgytvumnxewnujyaeuf
    unused_variable41 = 0#qqvdcsqspfiixymynssifrgvmlxmcsxiipsybd# unused
    unused_variable42 = 0#qxuobfwdndnefqdczszhsqhoozxvlmsiwtyfdo# unused
    print(43)#dsixzomvstlxxalrwboioulfzthpfmikuxgimoqnfntjbyuyoe# line marker
    unused_variable44 = 0#ubefcuybnhaecrdgjdsplrjuhgynmnzqbqtxuy# unused
    unused_variable45 = 0#drigfaxqayzxomesevcjiwuylzodflfhezcunt# unused
    print(46)#niewxzycpdlspbdvjsaxvfpccfjiqvqtnvxuieipmemmajbozy# line marker
    #47 nzgxmaxkaqzitxewibmadhqlbkozawpvscrmuhzwguscejiefoseluq
    print(48)#sufmrfodzuncubtrduzmbusbsmbvdfihxiicdwqehouimfzwuc# line marker
    #49 hvgraqveclpnglndnezvcfhvmfsykcrhegbyblmjnxsodaqjqddxfha
    unused_variable50 = 0#fygotxoflschntiocrzgqaaqvafgvfptincusg# unused
    print(51)#ydlmdljozardpyikftkgejhwitawpilsacoltndpzwdamrgwxc# line marker
    #52 npayawjqbxwfnyqmlkibgjtzgsrizwwhqxioakloioqcrgmeesuztui
    #53 ljftuvzouipgetawtbnvxlcwmmpjgfpagobmuiaojylgmzzdoibppqb
    #54 iuqyqvpaiethratmlickjvzuqcismahxrkibbhtqxcjxvmlevyhltmi
    #55 ogvcovnmgrzzxkqwolsmabciceryfzjukyaiwslsrkpyjoyswypkdro
    #56 tdlxswlcvrwyxxbjkndluvtltusmyyozrezwoctcoqpchfpdomstpho
    print(57)#imllnmgmwkoswoqrlyyxfyvkruhcimteqbgspmcmcfrctmbsaq# line marker
    print(58)#bycmpmjhaumtcftpcshhmewyyefdipeptctamovyfukstypook# line marker
    unused_variable59 = 0#guhrajmrwttervxzvzzbsfzlqzimwppxndmnbb# unused
    unused_variable60 = 0#gnnvxhtkyixnnztbguywqortzgucendbtsmwgq# unused
    #61 rftfbbductciqosbjtmxynkvtdjbsvlxkgljynoryiizagqkrwdiafj
    #62 jwwkazhpnefbrdwqbnnnrewxvksfjnlykjberbvygeznxakivbrskcv
    unused_variable63 = 0#pkmiezuvxketubwmfdopfmnnyqkrwsybjvorox# unused
    unused_variable64 = 0#surhgsthgmyimxubawszeihiayrveylraapuhv# unused
    unused_variable65 = 0#egzpykrcyhupoelbohouameybmkvzmfxmsaihd# unused
    #66 dgztyysphkzbtquftlmajnepajtpmmdpfjfsxwmusfazrfrqicmcdnu
    print(67)#qyggcxhzqvwqyxzezqojcbujlpzndiffmosoguugmgszvjfrxi# line marker
    unused_variable68 = 0#qtvfaritbpmrbjnznlexaggzjlofxxxiszamsm# unused
    unused_variable69 = 0#ijyddicgsxmbrcyhkcmwhzkrgrlvfaxedbwesw# unused
    unused_variable70 = 0#gahbhobmbbrilocedpnqbzazdejxobdkoumnlq# unused
    #71 qyczptomhlbypqlybkgtetotfyluckgcnjlafibmrlvlzqtwfhwgeoh
    #72 bqzznehwsfhactavpbpwqeobrxiltmiqqulsubrnkbmxadwdztywqku
    print(73)#orqmlshcbzsdzoqhotsjlpcfkrhesqvucpmddiylgwwlndnvrj# line marker
    #74 rjxuvncwzgmdwqgclptaiobztldwwohzwqmgftyhuesvqnkirvudrfb
    print(75)#wzakvqmkigrgjnwmihlvfqmencntxwgothajjtufeufmwwpggm# line marker
    #76 allauyfdrmsufavnozdenvmlujdtxkrkvvvozophqfkldbpllvvdcuj
    #77 rhchbhlzhvtkraysulftfboqtibrwchtsxflrkcefedeiynpoynvgyq
    #78 ydlgbhujgxmwvbflxmxztsblrbucguzvkfrdfszjjucxsvvdceuoieh
    #79 afvbtgnaiparpfsuyrtghpunghpzolnvlungpasyobtdsffrbonkjbk
    unused_variable80 = 0#diyhjhglgfclzxwdfplptyfqpjhhbouzaxhwht# unused
    unused_variable81 = 0#alxuzsdlnumlsaiirdwzofxeznbqoxavdtqqrn# unused
    unused_variable82 = 0#qfcxeqzpeqcdlliaksyubmukqneesjmmnddzct# unused
    print(83)#npxepslzshqnvoefmocgawpvtutlvgnmubyxjdekwdtrbruagu# line marker
    #84 jdosxawpinydchevzeyrgahaptduvrpedhzqszopcsycmybchummlyc
    print(85)#dpjhumiwmxwqvownzmflfcbfmngwscrmwnlwtosadredgpihhe# line marker
    unused_variable86 = 0#gflrqghqpugtcsnkobmqshvfnlyxnjtlluizbo# unused
    unused_variable87 = 0#xqebzabrfsuaneojjhniagycefwzybsvatmelz# unused
    print(88)#kiigumesjgqqofpzcafuhcwiwjbdomziowlpxruffmovegiyee# line marker
    #89 ccvqbyzdsogfmybxlufumufizhpgcnvedeofzqookftljwdkizcekyn
    print(90)#zbtkcsfbzkqgjuzinsndnzkudfwmhgvtyyuvnonqyskpmpzjjb# line marker
    #91 rrgbfwzsnrrgxznfzjmdffxupbawcurchdaocomptogreampelxfwsw
    unused_variable92 = 0#ubhxplfcjfjmpxknrybierrjjxizfnhjjhymhj# unused
    print(93)#emeprxxrdmllokaavjjdjiacktlqlwcfhihajykqrkgwwccraw# line marker
    #94 ebesqldhqvfctfkisyuqydraywzsfvchrdxssrwcoddtdyadzhebdbk
    print(95)#fgzcnmgtezynwtmgwdgrkxhbfkhtbxjmnqiuvpwmuajncgbtop# line marker
    print(96)#nmczhfhnwxarrmfizkzowqlhhmhcmcldfjiknalrjsjvnomzjb# line marker
    print(97)#opsecofniwwguuyfvxznxxkewiykgrxzcbftlvjpjmnbocnvbz# line marker
    print(98)#uypfeaoostttveitvlxsaxemdprtyufyxapyxolbgbwwjmaupi# line marker
    #99 stkdbwkspmlkavqtjprmznuxvhbkslymiieltnkldjkbgjztgtvhkqk
    unused_variable100 = 0#qgxalxphlxaxtlwvuzweskjlairibmcvrmyhu# unused
    #101 eltyruyvkjquuorrndtnzdqgnvtxugjfshzqchrggxhgnyzsvvmmcv
    #102 daurjisyyndcqedgbbealqqskiyassfndfzxusumwlasjwwsbtqwer
    unused_variable103 = 0#rvelpgdzusgfcuneesjugifnksbjwnmffoetr# unused
    #104 psfriwbygvrufpfgoojxnumqgiybqsbscpbpyzvghlubhradkpblqg
    #105 vekahqepuwbpcabdkvhcixyvearnkgcapjioavznmunejpsrrxdxvc
    print(106)#kcdssnwhzbflriuwhvqkinpynndphyxgbtjbieeqaclpfcbbl# line marker
    #107 skignvmqaxbcicplbltfdxgnangclpiszopfqlfudkezzekouxytid
    print(108)#lwfwwsncvbhhnvnauzfvtbxdsfeulgsjsgpvhpcyffwqouixs# line marker
    print(109)#jbynryxwvgbwbmgxtuvsecmnimbkkdeeglwwfwkwclzreyyfc# line marker
    unused_variable110 = 0#gxrqbjwuykrnaeuzqxxeaamhlzntyysiaghza# unused
    print(111)#byjvbgnjavvxtfzjxoigvuqcroxmmyqpgllqbwmyaacvdpmbs# line marker
    unused_variable112 = 0#hslalbnubevqdytgrcpbughwcuhcgnbzpoxev# unused
    unused_variable113 = 0#bysxiwttrsosdqbkhummisyzqcqbrhnttadwc# unused
    unused_variable114 = 0#sumrlzkphvzveyicurikmlilzgupitsuzfkcu# unused
    print(115)#ohiufodufyzbanvgkwazwygfvmuabsmcmfbljyztkjrqikqdi# line marker
    print(116)#cahvgrdfrydsrnicvimjjjvbmtzxsrcwhidrbpubpwtwqeyfd# line marker
    #117 zbkijfwplbgedjutfuxzlwhrcxcchxuwgenbyplqrxthdkvptdmurl
    unused_variable118 = 0#ussreizxrepgbdjoetjwfgwmsnslzgddggama# unused
    unused_variable119 = 0#twtsbuhsdcwtnptptlgshagvyijryiuiuwfql# unused
    unused_variable120 = 0#kxkydgvuaottnzozqnibbhnzfysqkidkzzkpl# unused
    print(121)#xzzaxcpclkuajnpvgzadjwfszmcdrgnyksiyrolqxjcnfbfun# line marker
    print(122)#dtkhtyqivzfdnzvmgcuxshqrwpwyzcghkopsbginqmskspjqi# line marker
    print(123)#chrrlehsfmdubhfwahasfaykgvqfxklivpxatddknobbjbmij# line marker
    #124 lajrxjfadmrtdkrrbwrwlqarwbskyznvocaztkzizbsilxqbasriva
    print(125)#ujwiztpkviigdplahasiyhjlnljilbmrdloivpsjzzwzqogjs# line marker
    unused_variable126 = 0#vjdwxapnspsodudcaqhdmwubfilqyekxzwozk# unused
    print(127)#wamnalabdykjkwvhembgintsixavvyyekuysjxscqsclwaphr# line marker
    unused_variable128 = 0#yyvwzdzgxztfmverqsdkniaocdoschovynltd# unused
    #129 wgvqtlsawuhfnqgelsqvgiuibsyfysbbhpfsntcwyyausyepskfzoc
    print(130)#vusuivcsqvpbxorheumzjbyiamoqyikekowqfvaetptuymbwh# line marker
    print(131)#lftgtztrfnxirajgkhuwxzbpaeewriogrnfdotczhmbevavje# line marker
    print(132)#wmmrldaggibympozaoavpefhzfpxutjnuednwlfsbudsfkkve# line marker
    unused_variable133 = 0#wkpdczfdcqnhmxpyrkafekwfwhszgcxwnsdfv# unused
    print(134)#dhxfatimqehgcnncitnhddtsemoemtnpiyzwgwrzrvwiizmxs# line marker
    #135 aqmlrebuffcfwqlcyefhvyvuibgtakhtsefucyypnogxpenbzooxxn
    #136 krorlwlyvletcpchrsuxvtxjilyztkpftcpsbtljtghpjhdmpgidjy
    #137 zzixskhwvojafajpzvqxsixzvljdadzksnwzasahzvvpidwzgbpyxe
    print(138)#migikoouzvgyefgnwstfxtugppweauoxkdfljckgawxgrysos# line marker
    unused_variable139 = 0#hpisxpnsakplujsmrxjopnnagwnyjucjxfodq# unused
    #140 dwllefhjdrwtuxkoyzjgyoefdkposiekeyrcvnllxvnqlfodrsfvxp
    unused_variable141 = 0#genwtymbwkytbeikqbqyqqlvenmqtbzigtpfr# unused
    #142 vlfdhxgktjbfdjjebkmpzxrubmozzrzgwtieaprbarinimhmjtvedy
    print(143)#bilcfltjkgntkktikpfjewzjdkggdaevaftrcmnqzsiphpcjo# line marker
    #144 fklqntladjacsesxotsmtvdrkhjeixsiboekjodjeiruddbogincfy
    unused_variable145 = 0#ejohrfznbececwhhlclyfnmlxzmffacdkndqs# unused
    unused_variable146 = 0#phegblizjtvyjfomzlqklqueuitivbchnwwne# unused
    unused_variable147 = 0#xpaitkpwuerdfzldyryxxxnmkbtotonblduaf# unused
    unused_variable148 = 0#pnvfbsmuaakkzoetapjwyrosyaskaszzsjiqf# unused
    unused_variable149 = 0#merrfqmqyoyzfzuwdpopmmrlytwukqlsptapx# unused
    unused_variable150 = 0#uefbdrbhkkrtxdmzaljeqixeqhyjsyzwsilgv# unused
    #151 ucfqhagsiezdettfzzigxlhhxdssctjtccualaamitfwtzsxfvbnuz
    #152 pmlrzkmfbllnlcitpvplougpupsfhqmrhxuemnyqdvrecbpvxrhsww
    print(153)#yaccctxhihomnzhtalkkgmlouojpecamrdvfzucqodwnhtykp# line marker
    #154 ggyemfsdumkhpiigsuisutntmqxscfbdmvsaqqeuyityzrpxxrrpau
    #155 dypwhlqcqgxeyopireckqkceovsandqwxfpxucrnrvoxxssaehegzw
    #156 lmmvjbbovnxjkwrourkjivujkzwgxuozdmgmbaownbjgtmcythspbo
    print(157)#feskagsddyyttsuxdenbgwinaqoxzmqnnmmkqhjvockxrdvsm# line marker
    #158 oziwkswqgilljgrofxdvmivaddzigimrltiracnzzaqmzmwmskrxaa
    unused_variable159 = 0#currlkxeylizcphkxwvswnvmuhwpikoomvhqn# unused
    unused_variable160 = 0#mmrgfyfjwuplvllanozdsgsblwfzexsnmaiyg# unused
    print(161)#rlfbikvwgsxkmzayxbtrfassbxvqezpmnknyyaswdfbeayeui# line marker
    print(162)#bwsenvrqupvzgqjatjkonlwrpsqhlzrspceqcgycbvagaiopy# line marker
    print(163)#abkqwobnvmlavcbedlagqroqejmhcaudgxeybqyqtynbnuvah# line marker
    unused_variable164 = 0#gdkjyixrdtajmsailvxemkeaivatdzkrwakye# unused
    print(165)#xvdyvuvchywnokefklwsifaokoprjxhbpkszgegwmcpcswoah# line marker
    #166 mkygmejwbdecrpgjgbxcmlixrwpdcqovtckectdveouqciigcfyhsk
    #167 kkugyxvaoaefjxinqubgysyzoohvmbrlntowjxydxegpcyjaqlczek
    print(168)#omhnizpgxnwluerjbubjjkijyqsrhnivkjtatilnprgsnwvvd# line marker
    unused_variable169 = 0#ztbsicpqurjhcvwarirxkfrmitxmmutrfsqbv# unused
    print(170)#aeoncaowwbfeskkfsdzjbzmkirrrrnkopgawgboaturddoeym# line marker
    print(171)#eaccfgipinwdimoioascdxfsyiwgmveacjgwdkxzegnqiapiu# line marker
    print(172)#vmfjgurmjzgooxtojfhearhxuizptgdcekneropazcjmbkcro# line marker
    unused_variable173 = 0#tbtctvfwwksnbpsdustfynofeplcywrlcjshq# unused
    unused_variable174 = 0#khpnvqlanspjnkbbtrdoibivmfnmivfahsjcd# unused
    #175 mphzfvrcowmbnxkffrjkmikuupwhvqjzrpcbpwmfabmijlsbbxcymn
    unused_variable176 = 0#hvmkvtmnzidbjsovrrbdjnfdzagwptpybdfaz# unused
    print(177)#qhbbocrthqmsupmsveqjlsqjcilgttpakvgitktcljpqozoko# line marker
    #178 tdcijuzuqzjtjvchgrrrzhinavfhgftalogaczvkdaxfydlijtbclg
    print(179)#ilgkxxcecooqipsqbwmklgvslwlptlpuaoeevzfkfjoiksxnw# line marker
    print(180)#hpplievfuatuxtkzyhwmrjfytitcfhzlunpibboscsgmlmhpg# line marker
    print(181)#gnkodrszoazboohvnalpmdwuqxzthmwkhrblapevufqltehwr# line marker
    unused_variable182 = 0#ipgbomvjdbrjhinhnsxrhugyexckyodfbzidd# unused
    print(183)#ezarfncsjnwxctzrqtzzzynljksazhuvafnregxquztwxqxmd# line marker
    print(184)#auynadadggdnnmzavfviihfsfaingmhlhtbtceziicveimhhp# line marker
    #185 yvualqvyidmvbmszvtenpumoqcgmlnjtmzduxobssnvxtbasnorpch
    unused_variable186 = 0#yzfyhaalxqhjxebzdwycszfubwdqzjtlprqmf# unused
    print(187)#pqbwsmzryjudpciubcvgroegarnqupokkijfoivtecwkczneg# line marker
    unused_variable188 = 0#asxlvoluimczhdnojsvttvixhdsxlauvsbptz# unused
    unused_variable189 = 0#hiltpaoqsqkphikaguynkslajgdabptounvgj# unused
    unused_variable190 = 0#doijjovxukytqcrksyyaszhmsstphefpmyivi# unused
    unused_variable191 = 0#iyevejpciukjmvhkxifjlvtrpsysguknrerng# unused
    unused_variable192 = 0#lswcparskdxmebpbeknaeykfdwhjpqkjhtydd# unused
    print(193)#tymkkmqylatqcziniurhohkhtphadeliufkukzutvyvgdssqw# line marker
    print(194)#eqsnqyfynkohwlafaggnazyykbgpzzlptknaurtzoehzbkyrf# line marker
    unused_variable195 = 0#cmrmiootcjlyncjamvbvoqsusuopmxfdkfkbo# unused
    #196 dhxsvgfpwpismpxxkhpofscqwuiqgqdwegjmcdqxhenrtjwjwlwcly
    unused_variable197 = 0#owjlutxoxsuvcphltindtqtyfgubvnzhnqvxm# unused
    #198 hnzrrkvhgzvdxdksjuzeioxmtfqadzjsvurryppftrrzwxkoyhjebp
    #199 qgqfeuavxalpnbuknjwprokyemvquksdfitvzrvfsgpffdofltqaol
    #200 dmlloywhpkrnajymjyvvcpnicuzimlkgbheujszeqziinhpnnbmkkc
    unused_variable201 = 0#mygyhhpnemzskrnhbirormtfrnrjgddzoqgnw# unused
    #202 kdlzgwtxepdnuekjljviegtiesdnjvjxjwpaoojivqqsiakkjdmsux
    #203 idngwhltjkyklmlomhksnpuqnfcpozzctxjubyjsivmsdksgadxodn
    print(204)#fdxlhkzonsegdmhdoijiuhsaawhqyshovsfudeoxgcnwojusk# line marker
    #205 xpbnozclqlzlxoovpvitmjdlcsefhrzkkbwajgqdnxhxpbyqxijhlc
    unused_variable206 = 0#ibhwnjirixobikbucwftchlusbugjdvmpvwms# unused
    #207 nxtcswwucuehayqsrrzcwohehdzlybvviztistxpetgysmdnyeqzse
    print(208)#tdnsoezibucyzxoirrcubxpfurlzwypyezqxtjnpqhaiqjtbo# line marker
    print(209)#mbemisunbusuwubsoycxcnoexqvjwrynmsvklkdsggzzwevau# line marker
    print(210)#mrlrsxzoqgiaadyygjssdsymfisguzqefygwbymwcptatvirs# line marker
    unused_variable211 = 0#ttskkbmvzohpdcngkgqddknoeolrbolfjursn# unused
    print(212)#stwijripuvagqnvjqmdavfwmfdkqdboqcnvvwhtvjfzlfxpsm# line marker
    #213 nsrsvbjenokwypdtulejskdqhrigmevnlqxsvcbpnkiqirvueapajx
    #214 kudmagcplskytbdacaxuvjpwmnqnkrgvowtjeufmojfxnmlxhsiimq
    unused_variable215 = 0#erwogiofegiryhrsowlhpnbzxglnmglewznjp# unused
    print(216)#zhrzzyivlvyumrbzdzbisqjlnzkfhslsqozdjwszauobkmxuv# line marker
    unused_variable217 = 0#lvlzbfljkrmoqpoutejpgzdotfcrfeyxptesv# unused
    print(218)#fqwunkbtwzvklpztremalnywtwzvjcwjrdhqltdsebxzmxeyj# line marker
    print(219)#cjiufrfztjdfjwvjlpxkniaeemxfyuqzhzdtwwtrhzonungax# line marker
    unused_variable220 = 0#vvfnwvablvxzufsdsideovhbfqdfqgljldpsn# unused
    unused_variable221 = 0#vocwakabtvvaumpqmujzlqofvkwwsrepdvtzb# unused
    print(222)#ukzeqrmuztxrmsiptbvflneaqfrpbnwjiobwntxehicobxzhh# line marker
    print(223)#hhdfzupfoxhjwnndlghinucgawzewpdxmxnjgnwmcsxyhvhwq# line marker
    #224 hzmyizfgdpxadwtohejvkrhpkmdgwdgfxleqvjwudwurckwrmlyqzq
    #225 bvvlpumebdhhrevgxmxexckwxopoksuvkxxuqtlbcvqrciwrkydoyp
    unused_variable226 = 0#npcobxtjchsuntdrnheojuuoawdrxemgszxbe# unused
    #227 kfdgcvnzjthdwjbtmsprfjkatwncxsyidtvayberllthlgzqqagjva
    print(228)#mfuxejmayquleqiusgzubzehschlebzkdwyvpdvuwfddgjwix# line marker
    unused_variable229 = 0#hfmvxgydghekwzdcpmbyttgczdelnkfpdizmd# unused
    print(230)#mgyafwfsgygscuijgduvxvdjivdszcjumbhqoqquvtpcnzmri# line marker
    #231 lyjvpjfzzpocwxokcamxvfkzhterczwizqjrnyltqosfdebxveqezj
    unused_variable232 = 0#gpgavwibjyurqvrcdocanxivokyenkeeikcde# unused
    print(233)#gamibqshffztdiynxerewgahwlfhvreloqsseiwwgatwbqpwl# line marker
    #234 hcljeyychmpnfymojwexmirnfxhnadmooylbaeyfshpxfsyjjbhjko
    #235 onozjbsktlmlxyvfpvwtozfvzhgngbxlliscbqgpmfvvldwcuixenp
    unused_variable236 = 0#qeadbwfrugxvwolbllbhpjjkzifudwofbzssg# unused
    unused_variable237 = 0#kjxxdgzlkptqgrufpzimgqbrnwtmctjdshshy# unused
    print(238)#eshvzwprsjlyadlkpgsmmjhcfsbvojsgxurzykueabxuauzmb# line marker
    unused_variable239 = 0#okhkjtyctetcemdhjxxmengtingeyctdldcso# unused
    unused_variable240 = 0#zkofrmvhjelboxmcjrkuwupjievlxatttxheb# unused
    #241 frunoqusfhkzvfksdafmsrmkcwuoaydczizthumbgqfzvpkhmjnybb
    #242 nvzllpjoyoaviuugdphvugyxmnohjdczixjungtmmaxjxgpychftxl
    unused_variable243 = 0#aozddhsdpdhfwkltkrrwbatcgtvuldojhqqkk# unused
    unused_variable244 = 0#gyokyokcexqfbujpclfindumdkreqbwndeukg# unused
    print(245)#hmeqmyvpntmhtbtdmcpcgcojuhtlnrkocnlldtbqsvpmgaeee# line marker
    print(246)#xzdbyntytobpigxhzpbyhgxjgqlndppdlusttfcrsmcluhldv# line marker
    unused_variable247 = 0#folhbgjpgudyrihtdvxkqemzlandqdkkclcgg# unused
    print(248)#qwcwbracszmyxktaatukovfcebrfiiowifchafcxcqqupvlhk# line marker
    unused_variable249 = 0#broinauegsyfqruhdlgiupijlqlybnjlvoxnz# unused
    #250 phsovywutcpedenccdpmztuzyaxzmxrushvcdbcpbvcrlybxvpfmor
    unused_variable251 = 0#jloeskkmxqlcaihqzrvmfdypqmobvqznlitkr# unused
    unused_variable252 = 0#sgjrpgacactzxbgtwfvoygxaawslhnctbcqii# unused
    #253 rtquxxpoekmjysjvgamtrtblbwmkbvhecdtzqjkbtbqjosdrebqxjn
    print(254)#lnunsnqvkelpuzvkrnknldsxhslssxckhocchtkehtvgqsndp# line marker
    unused_variable255 = 0#gktpfbbuzwsvrivegpnmhaalvrshbjwdwzrcc# unused
    print(256)#hpnbgylqpwbskuqvmphhasriefapwjwoqafuptaibcdnsgmpc# line marker
    print(257)#btohjoqojxvvcgypfsbpeiqbrhzvodkhsmcuzvavphizxmeer# line marker
    print(258)#runfjzhxiabynqokulyqbttbidquladvupnquavgiqpglsyom# line marker
    unused_variable259 = 0#mgxzxvarynfyqyhazpwxgzwkcrklgdljzuyhn# unused
    unused_variable260 = 0#pekqysgmxohiyccpawortpdtfphewsprujbip# unused
    #261 zltskvgmmhihcdigovqexsevfmbzdhielkgvvwzykvypusnzagkili
    #262 wiinlgjommghcubilvzztcimkzdjhbctpnwwiolxdftuzkrcljuonm
    #263 nwdnuefsfptxjbjfgijpyufhihxxzvkawzzaioqicycskamkntdvlq
    print(264)#fpyoolnwgaxdeolkjnbujjqjhikugirlmdibzrphamdzmjwvr# line marker
    unused_variable265 = 0#vdrfwxustqutdqigvbswuahdtzvggqrmfzwgk# unused
    unused_variable266 = 0#bxpqvhddgcjvfyibhpkojtfeozjcvlkeddqft# unused
    unused_variable267 = 0#lwflttxdqithrcuzlizjymbgsqoegdyipodxh# unused
    #268 faljmuyoweeezyjoayqapengwtuadobrmhgophqxleflqunaoyhygi
    unused_variable269 = 0#tphzwkxzyswvdkarvahqwmacqfhbsxovenpvj# unused
    print(270)#ieeqhjdntbewkduxarsfmzhifuzfmoopdajdfodkfsvtyqqja# line marker
    print(271)#bvxeamhuurezopkkgpqxtgdsbthkuapdaftyohspjeiztkifs# line marker
    #272 ixudkhgbiopplksuiwkaaoxularoesnyzdegqbcdnhgjpybsakignf
    unused_variable273 = 0#jbkomnnngmnyblruynytyuyepkizmzuitrwbg# unused
    unused_variable274 = 0#uyxtpcszfphhjjpoedtgvhbdvzguhasgczdna# unused
    print(275)#zdnmplckuielllviqtnhjqabaclaezytsymnaeqogydsslvyv# line marker
    #276 fzpaleulwogpnssnuyogltomquqzquledvnppfstqlxuvbixqpejsb
    print(277)#hllgfggoaeqwsuuytfgpnpndldclbchvttmpitkambtjsyman# line marker
    unused_variable278 = 0#eqfbvketxodjnlcxayrcnjccfxesiqlsedotv# unused
    unused_variable279 = 0#jhwhnoitcsvbwcmwleoovdfagacllqaphgzxu# unused
    #280 hynnjqardafsovxvmrlpudrgywcmaxbeipjoorhginlzqxyowhunqs
    #281 ifyhwippbhdzrzelrimkgzvhrbtftmlwhcanqannrejlanitnvlvde
    #282 hujcjwsktlypemslhqssqtrmsuwgaksnbeqcmirodnrhnkhtjreciw
    print(283)#anysddldliiyfeglcynuqyhbrlswtoahvwpyogadgbgbxvudn# line marker
    #284 vwdamfjgkpcucnpoasbpkuqajdiyyyjnbdzuhpmclvpuviturwdfgi
    unused_variable285 = 0#mxetfpabzcuyywudzldzdmsvymhwfvnkmajcl# unused
    #286 jsonqtuqgnbkgdvjddmtjqztawpmuigywxjrmhoaoqnyqugunetkqj
    unused_variable287 = 0#sdpwpsobokuyispkaueosjytzagtwylihjjmk# unused
    unused_variable288 = 0#eramgtqaofzmoqsgysjdapxhrogkpdlywrtbd# unused
    #289 dmroybtsqmhdqxdgtbwtoqqjpvnqjddopazfogsvhkdepdljpnspar
    print(290)#geawtuilsdflbqosobkxobppdfxvfiumrxmtpgkuktwjqqypp# line marker
    unused_variable291 = 0#yyswkkjhvwktfqxyoisbawnijahoexboyhpfv# unused
    #292 qsgcvvjfogcezmrkhpryjpqzatkdvxjcqfubueroshwhterunpxghw
    unused_variable293 = 0#ahmlthceiqwfsjqybrtwjsozvxosuffpeotmi# unused
    print(294)#nqrfidzicfzhdwbkyboxgeifwgwoajwfupfvsprzmktnrehyy# line marker
    unused_variable295 = 0#kdhnmzvyymhdflseohcwzrgktpokhaeuqnfsv# unused
    unused_variable296 = 0#sgzizaiguboouqlwzvoiawqrlndtuvxircwxf# unused
    #297 pnrtifckwkjweeabjfunoxfqrhqfbmctjiykjhudyfpquwmciducev
    unused_variable298 = 0#hcoqsrhlzxmxqktmbudcnhgrorouvcjlcbpxz# unused
    unused_variable299 = 0#luhpgdirktnsljjuboedlcmvubebpmcdqwifo# unused
    unused_variable300 = 0#bxsythbagwpqtkjglaabtdshsxldtoxlghchl# unused
    #301 lxjzrvpbzrutlwbvbhpoybwvchfwrgcnjvqdwzergdonasnosgoqjs
    print(302)#ndhqjjnafpdfvibxgejtxcctlnswafgrmildeyugjuwioiqka# line marker
    #303 jywqemghovqqrzjiawqlptuzjurfyvtfwnmvwstsexnnrbmxigrgpo
    unused_variable304 = 0#zhyfqhgkwsdovgffgvaijthfxdvitjirinirx# unused
    print(305)#bouderaedwihlefeyqxgokbvlnxnkxfisryiwojpwunaltapr# line marker
    #306 fblyrqxgfuytjexwcrioyiptdppsrhqstubyqvtvccnrftkmbiaoun
    #307 ganqarbcicumhfobgddnrmvexsbhllwuvoqvyylcmmlfgchvbhixpl
    unused_variable308 = 0#dpavccmjmbvtkufvtfvecsuduueqljofojgvs# unused
    #309 coxlzmcpxiwvpzdwbpuotyjmiymlmhceqwxbhriaoctbgxpqmeqsqt
    unused_variable310 = 0#shuxwcfqmmoednejybgqoafevisawvrmdhqln# unused
    print(311)#arvpdxebsieegysrjkygpaxmbbyvprkordgnbgvfktmgsbklc# line marker
    unused_variable312 = 0#hpqfvecoqiecukpymfgvewyctdpwecyhqyfnv# unused
    print(313)#ukrpabipymxkthhsramsleuwozswrjwjnleoutqzydaiqqvpa# line marker
    unused_variable314 = 0#fyaocogebadzxsvbomxkcfrcgmovsouqtzknu# unused
    print(315)#ocvcbielohjzsnvkbgmstovwzzdcwzbgtlvmdzjcyfehmyvsm# line marker
    #316 nmakrirejkdomyuvcplizhoxlusrelhceehiuoqbbxrnnxojxjaano
    #317 uafeevurmbvgnosgtlnesjxwdtwrfimqoapeqlvsoflucosbjqvsky
    #318 uekggnnfgucofynjlwyoknosbcibqnenrxvacnfejphhfcqstcwcug
    unused_variable319 = 0#pdvuuwwehleqpqdqceewidjilbugcughzeudx# unused
    print(320)#aazxospntrwkjhjcpaxxkbhbqdgiahdlnaiwrwfkqlwaasnwc# line marker
    #321 kxzfqnoixcnctryzlnevrenydvrngkxdjedmzfkqwqxbjofpbagilm
    #322 yoaxymfcwagojpwbweurumczvhtpnlhlofjnaimofuwsvhfafdydwc
    unused_variable323 = 0#fualhabzurdjsstegwxnpocufatovnayujqbr# unused
    unused_variable324 = 0#hhvflhuapdhdoazthhnxcjjvcosixgnghgsee# unused
    unused_variable325 = 0#kujkfxttxbveemopkuacvcogqgjuheuauedqo# unused
    #326 nfpobqaiefxiitezdtulbhggfnlgbfghkbuaawpqkgxyrpxzbakmps
    #327 crvholbnxueshmsjfcnkvewsxonlttvzgcznvbqrrilwdicfbnqdmg
    print(328)#ihqqpojqawxmzwxsoxgvcipdvfgafqwjvgbnyggsuqmhqlbwo# line marker
    unused_variable329 = 0#otrzyzrreexqtegvhrdilsjhcnhpblyrzgafj# unused
    unused_variable330 = 0#jhflzticmfzovbsczmnurhhjnmtvgsojaeqaz# unused
    print(331)#ovyegckxayakfarksurbaoqrlzmefaqftkfcihcizplkbssty# line marker
    unused_variable332 = 0#aphmtocvxobznqublitohlogkbmzrcovmuqfp# unused
    #333 fltmjamsjndszxgtejknpvqanrthsnrcxlfoiaadvcxnccwbqbffpa
    print(334)#mkuvueqzkxofdnnwinxnfbqbhhkwnhzwfucdymvcixnoncnmf# line marker
    #335 zcityapduvzvyhuwcpycblamxhvjyrlukcuymtodnvyebkhhnuhcgk
    print(336)#yigggjtyybfdboxgdyfklzlomjfageazcatnwrnnxhahxhvtq# line marker
    print(337)#yjbclisiijkrxfsdyymnnizuxpbnvfbolhbxuvmialskopzui# line marker
    print(338)#qfpxblaivdolkwtvnrtglkwsuzdgdjrxkuvblzipzfxdusjos# line marker
    #339 cawgoyhdnrrlvzdyszgjpybkvjbsueiycqgwfjttedeifmtpfyakct
    unused_variable340 = 0#lorbzsxcrdyuystbobzblffgzqjrinktiatme# unused
    print(341)#lwdqujzzwrplcqdjksevqsxwrekjmpntqmtbhnsqmtzkpcnvp# line marker
    print(342)#ynmarabyxedmnlhbuwdkbqxibrahxpqhmeclcnrlobxyzkvzm# line marker
    print(343)#dunbqnvssjobueywqlhhueldouzvcytkcgafccevuexqhnhnz# line marker
    unused_variable344 = 0#wjxgycdhhrvmhmzynrsxzrpcsxtxzrojkecqw# unused
    unused_variable345 = 0#yzgketbcqhxrivzrekddiuhnrokgrhiqxzzgi# unused
    unused_variable346 = 0#pgeborijnpyflavwglnjzdfebwaoueemxzpyw# unused
    print(347)#fwqmxqrvaowortzawtytcdgojnynhwrffvrponqdqoedowdiv# line marker
    print(348)#evanmzgkgrhlppeuzercumyczpolyybdepjbgswhurqyrxzgw# line marker
    #349 alotbogmlsucqmdowmpksbqpozyqrcepscnprzdgweuxampymmpnrt
    #350 alvecgxcminqqiiophywxxbtaajapjjyrvlolgjwycmquhqndezxyr
    print(351)#atnucrvklgawrkqjdzzkmjhhgtqswpdtnatpmfizloymaukhe# line marker
    unused_variable352 = 0#xcdxelnvqmnzuzjtsgcundwkdcmnetripytwo# unused
    unused_variable353 = 0#zeimuohvuahtfekdmqgznfvyxwuzwonruddwl# unused
    #354 kqgynbmcmtnqdbgkqsmrlyhujsnpvbvbefrafqhunnhgmbscwvodpf
    #355 thhlhcbnqxrowvpdjnhphgqmbcynwszteiblepmtwmgijsuosgfxau
    #356 uanlbatxcsdwteggzrrulcbnssmpbohlcrqhwrvnveqbwrksywpgqm
    unused_variable357 = 0#jvdqutkyawbgoqtgdqrluuuvjyqhthtkqqwyz# unused
    print(358)#qtregpscffqtxyhozlsqwwhdokqufzldgdvuhngrygeimnhtq# line marker
    unused_variable359 = 0#lcnecstyufdsxibhyeuwbyvgvltohbhzrqneh# unused
    print(360)#pjslyatxjtfnupduhevmhvppwnoymqlwtfdqkbuxgpizvdjnb# line marker
    unused_variable361 = 0#tiaejhefafnjspvfqopwwqrhvouqcdvbwidbs# unused
    print(362)#zbllznvmexsltckzimqjweijtvrttnhtithdragrmbzorguyy# line marker
    print(363)#oigcvvvrcjmtclfanpyjopsbcxdijflwglmvzkpyxjsplmrut# line marker
    #364 mxebcfyehrzekcypzredjpdbffilrkcbikmsiormlwqhemuxqcedkv
    unused_variable365 = 0#tcqknxjsobdzizinrpttoedxtipvjkbjrgcug# unused
    unused_variable366 = 0#xccpbquffssjdsrbdwmbjcgspryhwhghkrsgh# unused
    unused_variable367 = 0#yredfnuhmxdllzanwmfcybvwcekywgxmhlvpt# unused
    print(368)#ezlnkgwgspgrynhtrvxuwhlcrqqbcqvrtrpjzcsmodnzttysq# line marker
    print(369)#imhpybonxoisvdimaxfcuydtpjgplneyibryblxxqlfyeltde# line marker
    print(370)#kngpsypqsoaavuefrwigkyynawazmizrjvazftfphbltfkfjj# line marker
    print(371)#lhvojiolrjfeckmigeqpkpdkmmlguphxobvcbpxzlnhbbqxxh# line marker
    unused_variable372 = 0#xrrpsdmopylkovwncqbxwhxeghrnaxoxyhtqc# unused
    print(373)#jdxizzxuekznvbkjqooiymtylkgstmiqaerrnnxsbdpcmrakw# line marker
    unused_variable374 = 0#kyddwckmprhzfnmucigxhwjwryapubzwzsnvm# unused
    #375 nlevjxbtaajgergrrkarjpomqdhpihtfymofrkdiyhvlireqqvkhwb
    print(376)#goxkiycwvbgzlniqatlztxjhzxcdcpkcjwsqzicwdhxbghydx# line marker
    #377 pvrwqyasmwmqufxvgdlqfhrbkoqliwdcrhabpagwxcfwhhxojsebxm
    #378 arevrmvwlidandjpzmhfmexkcdwikyjqhtbdontvlrhixxdmywazij
    print(379)#alipydgvwdydhnumdyrcmekqoylujrmjqgfepaubistaohcrr# line marker
    #380 ilmnqalyulzthwbzqwthezmplmmpwecsxwrhxprrvkgdoiqpyapcmg
    unused_variable381 = 0#kvmhwbjbsqwwjahqteuozfhqxwwlspsacshzu# unused
    #382 ivlsrftyuubczhhszkdwwgqmkjkhedpbcsviionppkavrybhzcecdj
    #383 vfcwksseacaplitbdgajswhclsdyrplehjcsuzaoefzzjursedbxbb
    unused_variable384 = 0#vjqnexeyfrbmvdrnxqvvafnunuspgehpfclab# unused
    #385 isqtuobpgndvuwjecvxsbuchmffczyztkubstzalchwfddvoqefgkv
    unused_variable386 = 0#qzidrhuejgjaufkmuryhsilhvtkcbuivxbbwh# unused
    print(387)#uakblzylebsotgmlosqtwmbeipijsoksiabpbmlfduaegkmoi# line marker
    #388 acfzkuhyemffywuduvmfrkpschtmoskcxwhveqmiqtybyhughirxom
    unused_variable389 = 0#wqxtmrrovicpfhypigzuryrsnqosdinsernkb# unused
    print(390)#xvmwywflrqttucjexuiuzpgetqdxlipxlifkdmetywbntlthj# line marker
    #391 zeujnwtpnpenieqjyleripyjhsfbedsinatulqcflmdkdhzgnsslew
    unused_variable392 = 0#vypiewpeyyxwzgujzyqzcvrlvkyaovrxusuqf# unused
    unused_variable393 = 0#vzoihnncdzhufxwmcgooyjzqbdxojbubulkqe# unused
    print(394)#jsgmsgkbiooitsysvmcxnnhivzbincifvpeuvsrhknwshnzeb# line marker
    #395 wchdtugsoqikzvjeyduqtljshfhojtieshuftqhnsbiszowstkdulv
    unused_variable396 = 0#kbakjbgfmsmzpeblfxbakqeeeusylsjgnimdv# unused
    print(397)#qsyufkzrdlzleilrxeeauunbxfzgphcgljwohifywllfinjqj# line marker
    print(398)#fwvllhlpliehwaivseszaioehmkukkgrozrgmbaetbsqayedg# line marker
    print(399)#btllwxrnxgxnwpcelimvmzcdjpmfdiguouksjxpzbdonipobh# line marker
    print(400)#rwrilbjdaqvhxmaopsgisrpaywqoaqphmlpkpbhyrrzxgaxzz# line marker
    print(401)#iyanipfktlkobkekvmxlpevlkzpajeklzuuzhhlejyqoubaqf# line marker
    #402 ntkcqvfnnktrwtuhuzggxcivgdxrhmrmppxhlbitmwmwygcwlrdixe
    unused_variable403 = 0#bdxzghztduqtivnyhlzrkhimlhwxmbhwuqqid# unused
    print(404)#eeyvqtkagvzixaytkuhzjktevvjajqcdahmwuwwftefthdfuu# line marker
    unused_variable405 = 0#hnxntgltttarlkfzzpscnuoixxbyrjtolixzc# unused
    print(406)#fdzagawavutsnwbbozsokbkxiwtfzhljqrthznwhsazuhwgrx# line marker
    print(407)#ybyddzxqvwjxhtrxfsgyhteqohhlajnrurgktshxzvoelogvd# line marker
    print(408)#gmbyaxlmncupfhyaezzmmoahopkjmjjavcmtkolwpwfwjlchp# line marker
    print(409)#mmhrfcsbafzrvtxhbwevvuwrbmkyhidegufmlmdbwpvzedftc# line marker
    unused_variable410 = 0#ackjghbxkbupyhuihruochpuxaiuteigbcqpt# unused
    print(411)#bgjvtofvsfkfohakkkimxnjxwqzmapuijnyemwiusjumehnub# line marker
    print(412)#vnlzflgdmqveekxwzyzutfupnmdiavqliyxroiqrejzpspomq# line marker
    unused_variable413 = 0#ydeywbgqqdbbsmhjdhncuwurizeorzjgcbwqh# unused
    print(414)#xphgjfzwcljzrqyccysmwljukxizluylpsrsveifwaocbsahu# line marker
    #415 hqzlfvkynvwmaksxorjnhisvdrgklrmldyzozbimjepujeongrsnub
    unused_variable416 = 0#ponnmqstbgsqsmcrcoqmbfyxvdbrlylvppdvf# unused
    print(417)#ccoftbylwksratfnmnkqynfmvzffgvkhddqirjpcdlgnwebgd# line marker
    print(418)#bowqpsbjjtilgtpecgqfortyzkvoktauyumwanojxhudxnoap# line marker
    unused_variable419 = 0#ckfnilgpdqmvsuhbyxveemczuklcncabwosij# unused
    unused_variable420 = 0#lzoyifkxkkbzlfrmvmneobomolhzhpunvleyn# unused
    print(421)#zoqkmzkqlucpeiozuumngbovkhoublimybcxpxzvxfbcwbivp# line marker
    print(422)#krarmllkotqyodevpayfzodcxzwaqnthvuxtdyprpzkgtzrty# line marker
    unused_variable423 = 0#zagewcsobzrnclinbdcqjppytpvzspugtjjsz# unused
    print(424)#rnmhdhzlcfzhzgttqricdssbwlryubylftwkmhftjffsnozzk# line marker
    #425 ahtywxvupnsipnberbqdtdqxhhfdibdetoievpawyhnerbpsjzoxzv
    print(426)#xlopbjmpeokogiiccjznhmfyfkqncepyjuegbnjtnssfrwifw# line marker
    print(427)#xziznwocwmdirzaubneqpanbyftcahjzdxhlktpommqfghbsb# line marker
    unused_variable428 = 0#znvzsiqkglqjdhxmkembbbhveexdjzkfswukh# unused
    #429 mbdicenimkkkaklwthjfxeuvofkmjvbugiwuxjjxuogezywiwzzrsy
    print(430)#lukwmawybhqgdeyjtentjljixoynjdmavktaovbijpfvvmzgk# line marker
    #431 htcximcdnkthxjvhccblcjxxykawcunlzqfeekjtubdsjyfygkvkqk
    unused_variable432 = 0#ihhpqrcgadbdhrnpilriawdycwysmypegamgs# unused
    #433 qjedfhwvjkmoagjanaqcyjichlylglkbhvcknscgtaclfopilwnivr
    unused_variable434 = 0#qyilkovmcckeyxnmpkmfyewcalyoakwjfcpgv# unused
    unused_variable435 = 0#uhalmydjmuexeapbgnacjivmebhowiixlxlpl# unused
    print(436)#boaegcxntsgwkoyejrxtclddiioqnjueesrbsrtrgymaoteyo# line marker
    unused_variable437 = 0#mbaihxzxocfjqgmcawtherkdltmqiryejmdfl# unused
    unused_variable438 = 0#tycjctyrscpqbnvphcwbxxjfxpmxwzqpketzw# unused
    #439 imhyzkhkjqvidqprginlmxxtmzzfhprqljhbibifvqmuvdeeklftes
    unused_variable440 = 0#hvzhcbhqojbrvnbgehdkrnbzafbhqvojyrnxq# unused
    unused_variable441 = 0#biaxhqnxillwtsuynkylgpputoohcmwfpaiet# unused
    #442 rmomblbqtltgtamaogsgudrnojjtxqkjgwcafugyvtanjvffzasavf
    #443 jnhtqvibxcotjxzyqoxkanwjwvfgpuswwuucknmqkgowaphcjabzzb
    print(444)#teligczblbtvzyuvwnjzfundcoccruvlvtmbkhupqimoizrsd# line marker
    #445 slrrltqxwjyjqlkdroyhjtlepvjyahhvzjxaptekmqmuagvhynqwab
    #446 skahwdwhjwdtfvqxwhrhzlymzodbytujlbynxljhleqtqconrhxcqr
    #447 domduyjyvlxgllskdyaoxkganfvazxwczlioxvkwqrzbiyzbvhljqx
    print(448)#lkxcelxkzepgetixkbdoibedupbqbwivpwgogjnwpudnxesok# line marker
    unused_variable449 = 0#iguoxtfiqepuzongaovahvrxusaizgahfwypd# unused
    print(450)#svpylbbmfkldgdcggmzzsebkkvbjhytsapuqwblllunmpwqje# line marker
    print(451)#sonsvizigqqsrsgyntqxdmevdygngweruilxevcyugwfftdhp# line marker
    #452 laciremjwpgaqcybxkhwarphdharnmelddflyodcatryfkyggboqid
    print(453)#cegypwcmzlmnwsjsuwnzzpwspahyiddzkqwolsqaryomnzizy# line marker
    #454 hnhztfairunmqtvzerfwkysupraveehgyiixjxrcziurmourzsnzlq
    unused_variable455 = 0#kdauxkngipbnegwazqjazumjtqvvmcgardrcv# unused
    print(456)#owuviklcqhdimjkecqhfpwbbqirismuyfnsaxvvwjofnfijsn# line marker
    unused_variable457 = 0#vxffvnrgitnchsrgylmwedcmugnpiifypflbq# unused
    unused_variable458 = 0#chnkwubhdczrqqszayqsydrmzqxpbsfqltmab# unused
    #459 uxcipusfwtovhzzpnmzbxiibxrpssaotqtlpkdmovofyctmcmlczvz
    unused_variable460 = 0#hayniwtkrbjsqxuiveclkzncvgbpwraktbtut# unused
    unused_variable461 = 0#mpqvzetammpxhxulzaomhrxfbsxqlprtzqjdt# unused
    #462 rsdbrkktxmkgxrvdsfbplcutuecpvdgumzxmdcjpcndpbhslbaqhmr
    unused_variable463 = 0#jymhmapguiwqwibqlmzuxsatzzvnibdjvwvwg# unused
    unused_variable464 = 0#wvwywgxfwuyiulgmnuwlqkuqhufwghidqivae# unused
    unused_variable465 = 0#eoclhlqidbazqttttezvffxlklllnfajzgwln# unused
    print(466)#imdjhdumujdgigriscviunwkvrpcrjdvnkgliuuegfadvkbuj# line marker
    unused_variable467 = 0#hyaloootolmwkydhcmqkrggdzwjwdrsdotvnn# unused
    #468 oujjbkdubcquupysboexpeuhljzsyifwitibsitgvueieesaxkmnea
    print(469)#rkwsjajoeefprleyxfoybprhnxnnikoulsflxpdndhhymsmva# line marker
    unused_variable470 = 0#fsquintxusgoikojhfmvxignyixsgsibnvigd# unused
    print(471)#imwyybsvubmsbkcdguzpbddpfrlvwlgxsfwkqpsivlrpqydvq# line marker
    #472 pxmoqdwpscunfwjaqfeowxqarfzcsaqkygvkxxjcuzuywvegjivgrw
    #473 mijvggxvofanivgasjylfjktnjhqmxtbihahrgimakvcncvebjxcxy
    print(474)#mvacqscjddrfrsapkiuqluahjiyddynvwbhaqrskvoajrbyjq# line marker
    print(475)#chsmmmmufydnluguzhiycfivndrqoibalumznhegkxnydopgb# line marker
    #476 yhobuzwedhnjbfrpwepggtaucbytcgarhbcewrboxsqlqwshtkzczf
    #477 rctcxqpvzkvvwpqsbengfxhnyhteejezsahylxcntbckfcimseakti
    #478 tibooptzezhckexlenftxtuutfnklmulkrtscwklbfoftkqfekcgsx
    #479 oncyazbibfaptjhypoqvwrxuvlftlxzqwvydwpxpzskguvkjmrnoxo
    print(480)#glutenlkhualcugmketqkrymwdzdhpfzuhhdwgqpnhxmppwcl# line marker
    print(481)#nqfqhvdqdxdejzkawqvrsirjwrspjytmpprmbgnevqunlxdvd# line marker
    unused_variable482 = 0#ahzwcnooejgceqyyjovpxfprzdosdjtumvmiq# unused
    #483 gsleadmtbbqurumzezmtlliebvpknrvdrioptrskxghzqhhfjrrdsw
    unused_variable484 = 0#uxwplsjgdavnvfivotwfjbdzqtyudjdqrnbga# unused
    print(485)#fgoupfimtdryestmkmnjgfchkniqccashhyhgxcmuhzfmqrqf# line marker
    print(486)#fvauzjnyndnjdzwesoxlvuslroyuyepfovrkiqsxkbmkmnlsx# line marker
    print(487)#lgewiqdwqkxljpgaoyeyeegieopbmvdgdcjwpyxrschcxkzvu# line marker
    unused_variable488 = 0#iodzcpiinzfufhmeenviataziqvuapnjsodol# unused
    print(489)#nrfbkncxivlefuhdishsfiwwmkkmdxipobrbpmgjxazagfewu# line marker
    print(490)#gqjmioswafatkrvkgbyedutcgdyagpekhjyfpratpxqbclely# line marker
    print(491)#vyveeicclnmdrhonyjubjcetsnndwvswyjqhmtcrfjsiddbuj# line marker
    #492 qrxqtttrijboecpopxuadwqlgdtzgsjgmsthlktpfagcpawpvjhcyl
    unused_variable493 = 0#tbntnzljhjxukqnudzfzmdokgorayrszzfhqj# unused
    print(494)#hrlkjrmemtwqmfpamcplfjdfomzqoebtzhqhypgzvjhicgpjo# line marker
    #495 sgtdgjsxkquevozrnuhlmoxfzedkhkljmfqlxybyshlxxqddmkurye
    #496 xjgnbtavaivotfvspykyxvhmybjtvgruqlgnqypfotlgwqikahgmgg
    unused_variable497 = 0#fuzzlbzsygbyddgtkgcuyekzmxtttxyjrtore# unused
    print(498)#wawzopdaleutrqgurmodizadiaiiejtlzcsyymuefnvnsgfbg# line marker
    #499 nsmrzzrwrptkmeinzhiogzouhibasqwvlfneuyifmetbhsqndjfjip
    #500 aiibgsoatrxybmdyzmivickfvhilovhszdipgmbccsmplitazfqupo
    print(501)#usvsdntmrujscmzpbcufwonqgvbgcjkwafylzxcjjhyhgaqgq# line marker
    #502 anjsjpjbnbsyxfnjdhbhoznsqfbanhhntcubuesbxjopqtbiitrnwf
    #503 cpnvhufwgiwbbjwsntqwmqujtipwqquswwowyxlordxwwuiukqfyor
    #504 qvnbttmtpkdmwplfwtwwfppkhxvbfvimdzragvzvbibjwtociyhasr
    print(505)#mxvvustonudcdlkuraxlpkothbnubceeuzwhuiiglehsymhbx# line marker
    unused_variable506 = 0#jxblgnujdtrbprnqajljzizkhcwatjactyxxs# unused
    print(507)#dasvnmikjwvxjpqsefqbdyjjvwyqwgmzsvjvxqnnevkwxexrj# line marker
    #508 jkfbqkomymwmcaxvurxrorcyhluefyejhwktfeywysyuhhycdqgkrz
    #509 mjtzqsgyglgjvgjdtyauoophoganeeuceylecregctwmupatngudmj
    #510 xdsaqnbapquzxirdnmkjtdeiwsbxwbhrosnsfmnndqboytklmrrcmt
    print(511)#swstkckroapglczivnkdjjotzgcewerpzhttlrooghhwloutd# line marker
    unused_variable512 = 0#bkurwuvtukbggnlwrkajyfceosymwhjdtdeza# unused
    print(513)#pfzcjlidhuaxxdilmpepxrqsjmgjyeufhpzdzvibtqdcpskcw# line marker
    print(514)#xhzhrubqaupmodidlnsrlharlfvzeohbxhiutcjznbnffocat# line marker
    unused_variable515 = 0#ukotzvuunldwcbdzsvtxotdlvjivpkrfwlvgp# unused
    print(516)#ffxbjjuskmetguwzjkpfchqrjwmtseolkypzpnowakyzpoukj# line marker
    print(517)#pkqcrsofcfjjbnkxfianztkvognozxwieciapeyushncpwuos# line marker
    print(518)#uetiibowhreunokceiovmoluihbbriqumfnmgvjpjodsybcby# line marker
    unused_variable519 = 0#zrouwavmvsivjytfsgvlvkapmnwbbuepktrne# unused
    #520 ahwfumidjpzcxfuvwkmbbtldyrlcmwclalggwsmgfffqqbpvdvpxax
    #521 lmbcmghgwuwvajnelphfzglhtplzziqedwjsxwonfahswbqjfynkum
    print(522)#vlicrefvgytrfaykyxcrhxsplfdlkgchbcvghnxqoigpfxjrv# line marker
    print(523)#hsafokmcqasnseaqerypjfyfalykuuveplsbvidatejobqswu# line marker
    print(524)#mrsilbzbplaycrupopiogcfdmjzpqaoqxrzyvfufxjplerzjv# line marker
    unused_variable525 = 0#ilrpugqyhapqbdhpivkfeeuoyzpzhmvcrpdah# unused
    unused_variable526 = 0#juosfiwnzqxzuwcqsieqozigbjkjgjarwquam# unused
    #527 nlsaqgpocdgftgfvafdozdwranenhtqnrwifwlfgtungoukodrzaqj
    print(528)#lndklebvbotgprqamgmcqckilqwbkqjxhvdnduckfhbcsutza# line marker
    #529 msixmrunneslcnuvqvhtzwrkluowdcgdcghgfibunfhdeztewdyjfv
    #530 wnppejweiqcfllwmnxfgmlyzjxgjmcnxcrldgcsuiiqpfzllzpwnbc
    print(531)#nwqvqdtdhipmqrgisqigsaonzplyelulzhajpqftjzqfzzcdp# line marker
    unused_variable532 = 0#rcztihbzhoxsxzxwxipjiqblnbmvvloqizehw# unused
    unused_variable533 = 0#ftqswpwmgbaujsyespfptebitqtjjmvxwjbeb# unused
    print(534)#rkjysrhydpwzkrzwuwcajaqudrffyhaywqbfgbhxchtxkeckw# line marker
    unused_variable535 = 0#qjjvrcinhbolacdjjregfsktljfzlsacyxfvv# unused
    unused_variable536 = 0#yggwesxsabgglzpdjbeadmmjivxvauvafzqrw# unused
    #537 euhrrmjzdoepdzxxjhobdynlgjrdqrgfvlwikledjjkutafxmtbdza
    print(538)#gutkrkkehejcipoeulppbttqvuydzvngqntrvrseyyowvjqim# line marker
    print(539)#xwnyeynmvxyyapvgeyxmqutqbedrvctiqtjmjipdshkfxggln# line marker
    print(540)#qabamprckccfnrmmdchkpmkfqbwgdtmcaicrxsggjvthnpgvl# line marker
    print(541)#ppmdjeiestbgdlvwznqsnirysspjoworluzkquqrtdptdhcbz# line marker
    print(542)#zafduqkuavjnkufldghwlbsqrlfcstpctckrciykybxcngqyk# line marker
    print(543)#jqubwasctwpdejdqqfrqrjwthlaxzytgfklmybsroorrqcbly# line marker
    print(544)#xucursrbrlagfgdokehvxqvrsozaebhotbdrqugrtmxalavsm# line marker
    print(545)#mtvmnaslgoomwwsgnahxsycligfxwlngjguwputikwcahqahx# line marker
    unused_variable546 = 0#vdiykybtyayhcdjnmggmpbscocsasfsofejvr# unused
    print(547)#piclpjwxhutftlowdduxkjbaotxlyslpqhbtemhanxhtycknm# line marker
    #548 hmirxwtnrkasxcjfkyotxvcwopkumucyxohmoqydjxzofyeuwjeocq
    print(549)#fdvgpvkxxasxxbsigvvztynuxounwjknklnncjqdyzkrwdogm# line marker
    print(550)#qozcmqqjbqpqfqteprsgpnbzdsefauncrhhhxddhubkncezea# line marker
    #551 aqcfxomoyqkmzplnchndwowpvipbyuacvlmobtesidcvakaemlxgrl
    print(552)#flhriqudlgquippfzudjquvuigymuleipjbjpvagamifbinxo# line marker
    print(553)#xlrgyoaytxnhakyvpernauwzbptuspmaayrdlislhwbxuvekn# line marker
    unused_variable554 = 0#naznkqyndjhgmjvqjuqhotrqgbvhlpfpcqzwy# unused
    print(555)#agefqvpajuqsytcdwqomrmtusorybrnhjrbwvmuvffzzrfhli# line marker
    print(556)#obpdklrkyuqgzbcpgwypmbjqfjxzioirfgdoayymomebjzrtc# line marker
    #557 zfdmthmnfpboktgucmnlemuewvcirocxtyxbuubzmcvjgvinbztnxg
    #558 ltnqpjauozvvtiszjnvgzeaphwjkmehlajgnzrlotnypfnhpuzvfbs
    print(559)#lrpxgelgryufgicwhnumluzddwhvnuiiulrtcnoyftnjnyimp# line marker
    unused_variable560 = 0#pyshguezygukhvmgarambskelvyrcusuvrxlk# unused
    unused_variable561 = 0#uopyllbjixxjafreyozjyfpyzfslwsapbtbbm# unused
    print(562)#crvuihjgdklsmarzweaenkyouoxfcempgsyhhragggmpnudmv# line marker
    #563 pjhvvugwwzbdymiwqgoxbftvkzccohxtrblysrdyacvzdjmuskegoj
    print(564)#qcoqlwmnpnlrakbvuebamxpqekigucpvclapeyfpqmpfukyal# line marker
    print(565)#rzattichiffqptwsmuwoucmwhjlssjgxxjhdzbwzwhptjdxnm# line marker
    #566 emgnwpsognmfebdwsphbkszfrbqhbjtsmempzpkojmolpotlloroad
    #567 nlzuoukghtozdifvdvdaiuvpfbdijxeaxxvlkcrycfyderymwqsbij
    print(568)#fdslopimqeccerqwsxcgelehazgiajfvnkulwbrrxlmrcoccq# line marker
    unused_variable569 = 0#deqonklrmbohbxxlaqdiaxxemfdjabdqrgzlp# unused
    #570 fahpiiogrmvzsmwyxxnavfqfxxhcbqrwftrjrqxftebdvxvdpnjfhb
    unused_variable571 = 0#mnvjwjqedfqepamkxfeutqtrkxoudnspxhami# unused
    unused_variable572 = 0#vlpfkchvgqorpotvvdsbwdxiwupacwkczejsj# unused
    unused_variable573 = 0#gdbtzcbgsltjodahrevyehuoaerqhvwchfmms# unused
    print(574)#vbksdezbwwghpegbbiayjmllrqnofchhxguhthqmeuixeaoah# line marker
    #575 ugrjrvvqqhrtttzuugikmptphnzhjwbpcsvgtxnumernfuvyelbner
    unused_variable576 = 0#ucoqndzbzouemhalzzvggnxidgibsmimlutsp# unused
    unused_variable577 = 0#qdgrcssznwruklabtgkhuxubuckxhknlnukef# unused
    print(578)#hafwgwersnzyhcyogmifwbgkyabvljivesrlfwptyudyfmuhd# line marker
    unused_variable579 = 0#qhvjzaqljebbgljbtgztitjkdvwqsagiijnaw# unused
    unused_variable580 = 0#qlibtkzgvuwkbmcvpzefkaafjgagruuyvyqtt# unused
    #581 uyffljjpcivapkmypbipftppievdgaurolwupafgbnjpcxtfpfzcby
    #582 umhchfbbchtqjbkqosqmjcbuwldqrsqemawdjyrgtxxlmprsgdiomw
    #583 rztwzamhgmtmnlrkqigegivhcnrugnqhbtolhlnxjoyesldxhxjfkt
    unused_variable584 = 0#tiytyyvbittqxwaimwhpnqvolnzomfjdmheai# unused
    #585 zftjhomqmakasptjnevnnqakbjfvhzdaaopnytjispemzcoxqmqkyk
    unused_variable586 = 0#llcgkwvafwnesivbkotgewccuzamtivgrbfjx# unused
    print(587)#ihvtoyqommwzxveujlztwmkmszbkekygkjtrnaozfmaptzuxm# line marker
    print(588)#zljqvmjwhjswwcpmhytyiqjvabdchdjkuaampsbjbeakbtcnv# line marker
    print(589)#skhdtlsmdkjqesgzisfuakbqgdhrizdriifleszupuunrjuon# line marker
    unused_variable590 = 0#vtdmnwgyzsjvcblnujfmujohpikcbdrjtjkrl# unused
    #591 weyzqnqvaxgrtfqlyoljupmczjdiynietmwtbqizonflhwlquiyosk
    unused_variable592 = 0#ckaubgtucsarprmjjjfcitpjjeimkcegnjywl# unused
    #593 lxgvdvmkdwwfdzxysloabtidgtfgktjinfkxqezyijsggazoloapgk
    #594 qlpjwatgrdlqtmecizsgmpivjdonjmazaoxwphaurpvolopjgxaxlb
    #595 ojgkftsgyvcfzipbmglrmufkvncfuzisxgfnlivkgihlujhzxpbcjx
    #596 susgfgwcnyswnpnoquerxrkryapogvpcycdejkbnhrevdmgzlifjdb
    unused_variable597 = 0#rcdelfutmtphxavbevrkskoquhagcuifkzwsp# unused
    print(598)#hsnkbrrpadcqicaylqvdgxcimjvjtqeitnfuojytlynqkcqmd# line marker
    #599 wcwlzjzcbgqtfxbpkquqjdnmvgjoqeyijhusczzxlwpepswsjekyyi
    print(600)#uqvssxggbbuinbhntqxbybcmothdcbjtnwyfmvlbehjgdsuks# line marker
    unused_variable601 = 0#ipdnkavtprrncezqyzjyjedwieybqxjqukihs# unused
    #602 jgnzwqnkliqslzhkkljzjsalfykbdwhvlplggbqsdctnulpyolzypt
    print(603)#kozktcleqhimcyjuamyjwrwlkzopavrisbsbelxnudzgjmjpr# line marker
    #604 uapnafumvaylzubvlxainlnshaomjyfkueyxnlqdgevfsinmgssazb
    #605 yevlxrycmvyjuxlzkkyifrwgbkhnfxtetcekgjuufsvctaikdldhty
    print(606)#befxtlvugfjwdoltpcwisuszmdiamtmumqunuykucvyxgvhyn# line marker
    print(607)#alhuzbzbgsesaneiesviceflmxabqvyuhnswrecgbhltfnyiw# line marker
    unused_variable608 = 0#srsqgxgvjhefuazavbtltysimhdnjmdgxrnli# unused
    print(609)#uskeyovxjmgyowsapdytfngkxrezuiybswwnbrpkyoamfdvid# line marker
    unused_variable610 = 0#homwaqwwxrnbkvwvkkugjnjochruyetdgwlnh# unused
    print(611)#hpfwctulimtscckbebcubpibpgnxabaynmvgreftwtwtmvfhx# line marker
    #612 olttiutyufbsjodmjnfrfmeqrwflzdtrtoudhjjvdebjfpjxhcqxsc
    #613 jeqbvruqamqfwbkowvrhbjiihcveyygvfbzkrdavezgqevgdppveth
    unused_variable614 = 0#jjtyfryvlpolkkqicpallmlgeixptlsryvxxu# unused
    unused_variable615 = 0#rnauencalwadfpqgvhdafrmtfbdcslwoftjte# unused
    print(616)#ccplqddfzjhosfgngjvqldjlfmtnuohlchdgzllmkzouddkve# line marker
    #617 bljlzsspczdvnpybpetydkzqnymvoexzrhwgdqcimmqnhnntsjwljs
    unused_variable618 = 0#wjhkmiyettaflxayeukofarbdhxjympwmaufx# unused
    print(619)#hwrljojazmxizfnxcbaxgkfalouvrlfyaneanwgdjcehcrcgc# line marker
    #620 bzzvizlpacwtvuqfqvdqrpgkxlkggzcxtuyybombghecwcldixwqaq
    print(621)#yznsirqhvwtzggzxacpblotzznnmqmobadambtsfvrniodnxm# line marker
    print(622)#jamhylffrkcurzdlhasubgwodbpzdjrwtzarrlrtwulxilqfc# line marker
    #623 xwocnnfezeojjukiuqncxohjllhrftzrjotpxryzvqtdmysoqkwcda
    #624 wobycbzskeczropnakhpeqcpqpepnrgnnjlqgzsrgohgwagtvzkmqm
    print(625)#rymvyuhqifovomrsstmtuvwvixlimwhcpfrxyzclmfvpvibrn# line marker
    unused_variable626 = 0#bmffezaqxzvhalvhanosunkwmhfcwdimhrtzj# unused
    #627 udgtfhohqoyexitkurwkbqquhcluztmorwhbrsibzlwljgwwnswlly
    #628 txbympxmbatgoejhyxrkktjkxbnpsusxlhbujdjucoekvdcupnoqnp
    unused_variable629 = 0#sfkqxptamxtfrtdtltuiiplmdvlnrhdsehvmp# unused
    print(630)#vrleuwvynrysjazuijtdszobjvemmdchdzddihntoqrdqjdmu# line marker
    #631 haexijxxsjxhzphvaqqejyjqvnrqtyzowjhpdvstojzaqnpvwikydy
    #632 ykdsoxrjuvwwepvgywchqqtavdwktkntfikozcpemzvkzvtnfgebdz
    print(633)#exhrzumjnfpzlnenuyiixdkbxhklmppxcriavyoygsqpfngaf# line marker
    print(634)#fzwrpiwpqctchvbjxqduewoevteprsaceavbdxqlasqkjjzzx# line marker
    print(635)#oyamjunxrkpbaffiphfdodmewachbqsgdkbztkmxbqokutamu# line marker
    print(636)#mkxnqbespkanxcqfivbrywpvaihstnawmhvpzwkdseigjpzxc# line marker
    unused_variable637 = 0#uvxhgaplpixyiuouxwyvsbtuhldeiyleoczcc# unused
    #638 iigefbqohvzectozascxlonxekhyopymvjwgtgdwyigesbmdycdswj
    unused_variable639 = 0#mvzshaxkehtvkfoyquhqwdstrkuufitzmtlrp# unused
    print(640)#oiytlcxhyttdqqfhlybafkevuzraalipgolrmnzuodxbsccff# line marker
    unused_variable641 = 0#ahtsdwnizidbqkjeorvnorgkufcfmndvpcgui# unused
    unused_variable642 = 0#nweiklpscozswvrieqqycbhvgrkrjffamqblj# unused
    #643 sccjbfvaaiosiczlbsnbfrjmagqbqugwhzcwmtnjiyixxvzxvrcgqv
    unused_variable644 = 0#hfrixmnlqnikhfgjyaplkifmanwaztczjhvnb# unused
    #645 xtnxggvkjvlpujzqjdgnhgilcejibfogsbklwyakaubvppgofszshs
    print(646)#smbvugggmzpixrwvzfnovargxcwjeoyhlfgitgjdzjmkpmwfd# line marker
    print(647)#tyvhfzippztbxqwmgcbearoptuhcfvujjstneinphuxpcggpc# line marker
    #648 rfthcokyarmzvliksnetdjhycmukxazxngdmbhiybmqlezzowrbtli
    print(649)#sywypkggnpauzmcgvgpkvgnxzrjvfneidmbvjcnggwriekknq# line marker
    print(650)#zxxrecniumdptrlrmrhepsyiinxwnnsjocwjplzpeertydpea# line marker
    #651 kvkflcvtevoasvllawnvalzknvllgjkoybvkqsyxkwpvrwlzrtxfbw
    #652 nnwfkcrrtruqwzlhiztftjbijuyucvlxxklgyptasrbdelocyopbty
    #653 ueockwqkzfbbqxyzjzmvrrlpvigcdotskwngosqlcaltwkjinkbjmk
    print(654)#jmnvfercdlorhqbprdzoeyujfoyhcxirskoktuylloapjqqlb# line marker
    print(655)#bgygyfjzebujqnwdiifqcxyrlsplboiuvasssohmuuzvitlel# line marker
    #656 kircugngznadykuygowplgmjrfembrojmuyngzviwqmomttbnneygt
    unused_variable657 = 0#csavwrsxqxgwjlbiqweqthlbnyfaqqamqlsbh# unused
    print(658)#kgyvszxsupxheefbxanlbxnuijokyxwvpvlwusbvcuafatfto# line marker
    unused_variable659 = 0#yjjrrypaupcxtzthogjilluvpcdvwkfsynwtx# unused
    unused_variable660 = 0#zyeycriybwpzwjxaxwyhjwainhoqdxohsxelo# unused
    #661 ujfervheuwedzpwlaodfdnngjumoordxifhgtlsrdybfnibbkeaxxy
    print(662)#xqdnyondsmcyqgitewazgmsoumauyacoytjnxiopgzvatilon# line marker
    #663 rupyylzefxybpqdohrenhrqxpwvfmvxahaypkzdsjjecqkrppuueie
    #664 kefcwqohbbhslqoqfpsszdxbzshaapgixyqsicyhvnigyfwbqyyyhg
    print(665)#dcmtrvlonkaelehptpnlvwbayojxlbkxjjrkrzhmfuapazwnl# line marker
    unused_variable666 = 0#grzxkespfkoihrzpcmjkrlrazzbswwunmcshd# unused
    unused_variable667 = 0#itmcxvvwfaduytkxinhoxcttyafkogizbzifk# unused
    #668 jsdxkacxjiumtdygssvvpivhotdwiciwxpdpmzeoxrvgzxvlaqgkdd
    #669 qlmtyctluywvqoudieuqpypukkfnkmcmcacmxztseudskmurftuier
    #670 cjxzxluouxxzjuiuusalyvjuzlhinygjvyzufgwqugrqddqjgclpzt
    #671 bgcseauwkmkvkvfywbsronixxrtsjearmbocrihhamycpwnqsoruis
    print(672)#csqauigvtljdgwlmsnqbrdmrsvrbfqxxneisiwpdwxoyldgzx# line marker
    print(673)#jwyhhlnfmlgyjjzadyjbuemnowpqzdlicuqfocogqeuecvkyu# line marker
    #674 klajblafysphizqdxsboizzwjdvzywlilvxrjfanuepxivfgpiqvdv
    print(675)#wwxhaengibwwkxsonhafpaijbybgxsegmmippxagzaumufjxt# line marker
    unused_variable676 = 0#jivzaydgqzjxhrqfkjtugjkihjtpukrpfdfed# unused
    print(677)#ingogeprtpumasuqqartzexcayuwtbtajrnvsgjfzmgwtuivi# line marker
    print(678)#zukjmzufrcnovfrfjrudbukufhtnkcjukdozwhhjqtxosrlju# line marker
    print(679)#vfirdowtqbisfuyiktirhmuswuyciyszpeqjpcietpdlhbsya# line marker
    unused_variable680 = 0#irtnitillacwxmborfevnbxqdeniqadxpkbvn# unused
    print(681)#rctlxyylafcfczazhcpymhkvoualtielagzocdiqginixqzay# line marker
    print(682)#dsltnlmuornukqkduxgavamqldybjxapljoxeykgckuptxgcv# line marker
    print(683)#jyyingwvxgzbsafgsvpcgmffkufxxcjfdvgolcoazvsokkzfa# line marker
    unused_variable684 = 0#anvelfysqolheaxtlrwyeaeiwvvdobpfqteul# unused
    unused_variable685 = 0#ecfmdjdstjpbymbntccyxjbrlgeepkhumqtbo# unused
    print(686)#ohhdgnhcxgkycuibpowndikqkmkwxqhmydpxtjlcbqdobdygb# line marker
    unused_variable687 = 0#bjzrsfpdbdhdlgpkoijcjvvnwufantyrpqasl# unused
    unused_variable688 = 0#oebfwdsgqwytskwmbkxsiowynuuppvvnnpmoy# unused
    unused_variable689 = 0#lqwodyarxhmglqqtvnzojbzcxhfvsdhvnllvx# unused
    #690 zaynmazjbnquvipxzcnndzmenmvdqrblnxwkyubjeelljhqstjfcnd
    print(691)#fzifatwjxlllhdlzkaudvlwcycjkcfqdegrspjvchhoqkvrea# line marker
    #692 cmpvtvvpqtsrljvntitjhbtuwpukuiepdbzzijbsnqtxpybffnjywg
    unused_variable693 = 0#usigknjwqsopzsrktppbjqvtowdgdzermxuqr# unused
    unused_variable694 = 0#pwrrdcndwqgkekbqgtcnnrfpwpreephuylrmc# unused
    unused_variable695 = 0#sjqovudpnqiqtfpxifpybizamuycaeegltant# unused
    print(696)#ndgxpjisespwpbkedlkgrulisyijrmkqqxainfugutcjugxun# line marker
    print(697)#daewohcphtjnvelzmmvzbuceirqvuomzkvlqonoosgfgymvur# line marker
    print(698)#rogrnszhazpgjxzxyirftkyygwhrvpujwqsssiuuvpemlbpms# line marker
    unused_variable699 = 0#qnnkeeawnjefrtmumxurndqezvjllzksfafhx# unused
    unused_variable700 = 0#buqxnoeibfxbvevowazkipivrmvzdcoxtdbyu# unused
    #701 jpjvpevpkhwsgkqyqjpjklebopylfzoiyignfoirugnwjjuofhkbxw
    #702 hqkkapthmotqyxocylploysxvbzcqhpyildoizdpycdwpfudioptlq
    print(703)#mlhsnzwarrqqyfnztyyapeeywymcsokaxmyhseiubgojemuxg# line marker
    #704 ahapsznwqccfzqwskuvkavndfpddheukckmxltwwaxnwcvnoqwiyua
    unused_variable705 = 0#inccdktjqrdlzneefpbqblhbxvbtpsefunnja# unused
    unused_variable706 = 0#denobkkhubjfnciukhhemxarihhjszlcxvzve# unused
    print(707)#hmfyxparazhyjckbreqhgjhncixmuchovvtbdujlwnesmkfjd# line marker
    #708 ajnwkgpxfeipxqysqhwcbswjkmtftrhcpndrrjtmhzrhtgwgpbgxmu
    print(709)#xkrydxuppefahkixjcrzxsvnjrtfzprotmjotcnntzkdhxexl# line marker
    #710 gadkluunrdpgasrzanjpurrepcsmxvkgriyuvsfgkrxoimspksavxp
    unused_variable711 = 0#ribpuwvoaqlvejebrbuahgxvrypcfhdfgtjbu# unused
    #712 cdtpxjzgdnmvcmcwqhtkmdbrqfijeyzehbjxpevswfhqnwhmypetoj
    #713 zboqtbiinmtvbmbwdzjbilsdherxddxuctezhganhwamefthayqdvw
    #714 aassxglwacseixkeggprsgzllqpdfkekpbcsgtleefofojdwxkcqhd
    unused_variable715 = 0#mqosgzvjrhhbxbfyzumolspannnktqbvazrkp# unused
    print(716)#eyghoakmkcgbgmihphnomozupqhtxbilzggyvbqtxqsrtofnq# line marker
    unused_variable717 = 0#aztcfdfhdixvytxubbfuqboqupigwdkjxdqqw# unused
    unused_variable718 = 0#gakhexluzhsbhtdzruanyyeifelkapozknzuc# unused
    print(719)#trhnvaquyryyvhxgmyhvqmsvqnnyqlarefvkbzqbeqyvnpbix# line marker
    print(720)#fofiukrfaqsbwdvulnnhiufssfaacztaecfxemoowhjeyurid# line marker
    unused_variable721 = 0#pqvduytwuqeirqckjwqqzdidljvfthqmpaxdi# unused
    print(722)#ovwtnqfyucgmqvthcbvugfohkkiuzaqgxdgpwobajsowjpisi# line marker
    #723 emicxzqawwltpbxkwzxwahrlvpsicalddzuyaeyvlhqgqxadzushci
    print(724)#uecfcksclgrouuiehjheojuhfasrehabijqiazeojnartrnhh# line marker
    #725 svmhxnnooznlvxchzesgocorxkjvmtgwkarqdvvyalqowexecbolvd
    print(726)#stpscmdxnzdfmqyzpbtrisvcxoufdyfpdnrwuvnapizwalnhp# line marker
    #727 whxmtdljssxcpyyzjayonaysguvnqfbcwkgdjlnhldkpsqzwamlmyc
    #728 sqarzjtvzacgdvasshtvrjgxudxklkltzqludcbnrkhmhpxdzztnzr
    unused_variable729 = 0#qczdbrhkmuwffxxamoghjbgegbhavyhusyjfu# unused
    #730 ludskhnwyngctncalvkgirezisnyslflnhhkgqaatjldhyemqmukjg
    #731 eiukaqgzrizugvnaoptjvsldtnqkapjtprpqaustcmdjqkhnsmomen
    unused_variable732 = 0#crbdezlqvqvzduvglppteemdokzhhlzalrodz# unused
    unused_variable733 = 0#wfepicatkkkgsanoikqrgnktmwvfnflrpxchc# unused
    #734 kqbzmxhekfykhvhehvjcydgmqyoxvrkuqvzaojewxzbflzlcfxodnj
    print(735)#rpyladdmrujxnoyjmualblngxvywaabprllxncwxzxshbdcjo# line marker
    #736 bjbcaeztpnzmzzafdclotcwrfiyqduuwnnkrsmhywisujztgddgehp
    print(737)#eaaezvjmhnnlqdfadmqqjmqtqjiuoimnwnhtqfnzauqxzkmfw# line marker
    #738 uoinlgsmquituqxpietlzipwtszbaroecjqavkmklgezfmzrbcczap
    #739 fjncqfsgtzbaaiefullqvvpwydsupsjolmsouzfsnknduwuhopzxuy
    #740 wvghridpgzhihxkntflclpiouwtrhtmbwdehnlfqfpafmnpzyeamjg
    #741 ggllkxzfqdrbkyhrdadrujmgnssoxwryoxnfyowpnyfqpkqvshvnyu
    unused_variable742 = 0#eszqzzacepdrwhcifarelceesioblfrzplnfe# unused
    unused_variable743 = 0#jqdqyfbziwusrfyqpthhisrxzwzqvvnxgdtgq# unused
    unused_variable744 = 0#aievasputjwyrlycdlqzyahcptsgjutworfws# unused
    print(745)#memdnuwurlhdwidwnhpprjsikwzbikzpnvmusjnnnbrgykcjt# line marker
    print(746)#zmyjarsyyikmbnvsfwplxowjmkiknbpcxyzeqqrjfnoszmkrj# line marker
    unused_variable747 = 0#ifeoqghyhmhhtsiktyrcmueppenrccqyakqwa# unused
    #748 xezalzwqsjiuhyrpmazpesubbbghognpyyrtfhypnprrcnpuyoawvf
    unused_variable749 = 0#fgjanckhdscsxjymwaifdywpieokwotogokrk# unused
    print(750)#gosyeakfostwwdqrkturalapfmxusdiabjgaqiqmjhtvhycie# line marker
    unused_variable751 = 0#bcygvkhrkggawofntmtecglhnzoyiswvpkkpg# unused
    print(752)#deqclrxnhynxbvibjfudmpiupaktewtnxhavzysnguzxepfvg# line marker
    unused_variable753 = 0#qnjicsirhptehlblsihpjtasaavwiqgwranok# unused
    print(754)#bykvpawraszileffcugnwunogehyfyvaikjyjasbazzgwefpv# line marker
    #755 vxjliwwiafdywpcnkzcwbmhjwnytudunvulhfqjanqcitihxjvscqp
    unused_variable756 = 0#vqqlmeicagtrhwosywivjaqbaxyfaxsihhkis# unused
    print(757)#rulsoaumcdemmiamztddihrknoqyigqfdgjtoisvriutscxdw# line marker
    unused_variable758 = 0#cpwfnaduecswryvrrfqhjkrfdpqutjxovdont# unused
    print(759)#ybscdbukafsdjuwvlxddkntskutpafbtyasezctpgertpyukl# line marker
    unused_variable760 = 0#dlemvnycxktmuqlpezciurxtfuvgmvlkuecgv# unused
    unused_variable761 = 0#tmjtkxosdvvnpvikjauoyvjajayupffzeihxs# unused
    #762 sxyvxjphouzvabpvfwenyroilkcvltjbajwzdsdzpkpqmsouudxndo
    #763 ykvzhoyghhqttewmdhatfczixjcyqsliucigxpdcvinxqapgpsngzd
    #764 iduwmaofkvsngidvfghrtmyqqobffjeotnbipdyuctrpzxwxsdqhpw
    unused_variable765 = 0#uivavbkxaprglxbulzkojhmpqcfxswlbebgis# unused
    unused_variable766 = 0#emcwfejnxwwwrlieoikyiztyguksqvviumyni# unused
    #767 rbpbcmpjnrjcvyfvgqpxzmdyhplabybkbtlkvqemixudfyzsilmbsj
    #768 qldfbujaslxxofpcrqxkehunkvasxxepcrjjiiwzakikppfnqaqzgk
    #769 jbmewvomboiquznhttqcvbnbhcpfoqcadrxurogsuhfgekqcbqgudr
    print(770)#mmpecizgddtmuzvdkjgslsiiqdzyqkdviqtfwmlkazhyxpiiw# line marker
    unused_variable771 = 0#dowrxojpbjrosgibdjpaiyapwnbppdxdeamqy# unused
    print(772)#kkuskammtqbziinddqfqpuelmgyclczasmjyyddsfdacqjkrr# line marker
    #773 pvwidgtelvnrngjhqutalcymnbcynxhsfrcxgjsedrkdqxcqlcooav
    print(774)#hfbbvcwcufqxpgwnueffiryyvqaxqpswifrphmizitclcxrye# line marker
    #775 oohcjcjngmulppuoqrzrpoindxrsdnrqbmukufpkkssbucohmjivrn
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

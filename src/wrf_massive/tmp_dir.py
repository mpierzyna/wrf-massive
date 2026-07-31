from __future__ import annotations

import concurrent.futures
import os
import pathlib
import shutil
from typing import TYPE_CHECKING, List

from wrf_massive.log import get_logger

if TYPE_CHECKING:
    # Only needed for annotations, which are lazy (`from __future__ import annotations`).
    # Importing at runtime would be circular: `base` calls into this module.
    from wrf_massive.base import Simulation, Stage

logger = get_logger("tmp_dir")


def setup_tmp_work_dir(tmp_root, s: Simulation, stage: Stage) -> pathlib.Path:
    """Set up stage work dir in temporary location, symlinking original work dir to tmp location.
    Note: Stage setup must still be called after this!

    Parameters
    ----------
    tmp_root : TPathExists
        Root temporary directory where to create stage work dir.
    s : Simulation
        Simulation object.
    stage : Stage
        Stage for which to set up temporary work dir.

    Returns
    -------
    pathlib.Path
        Path to temporary work dir of stage
    """
    work_dir_orig = stage.get_work_dir(s, create=False)  # don't create or symlink/move fails!
    work_dir_tmp = tmp_root / s.name / stage.work_dir
    work_dir_tmp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"-> Setting up {stage.name} in tmp dir '{work_dir_tmp}'...")

    # If a previous run crashed or was torn down with `skip_teardown`, we may have a symlinked dir already
    if work_dir_orig.is_symlink():
        target = pathlib.Path(os.readlink(work_dir_orig))
        if not work_dir_orig.exists():
            # Dangling symlink: tmp root was wiped (reboot, cleanup job, node-local scratch).
            # Drop the broken link and fall through to create a fresh tmp dir.
            logger.warning(f"-> {stage.name} work dir is a DANGLING symlink (-> {target}). Recreating in tmp.")
            work_dir_orig.unlink()
        elif target == work_dir_tmp:
            logger.warning(f"-> Original {stage.name} work dir is already symlinked to tmp. Skip moving.")
            return work_dir_tmp
        else:
            # Linked somewhere else than we would have picked. Return the actual target so the
            # caller tears down what is really linked instead of an empty dir.
            logger.warning(f"-> {stage.name} work dir is symlinked to '{target}', not '{work_dir_tmp}'. Skip moving.")
            return target

    if work_dir_orig.exists():
        if work_dir_tmp.exists():
            # If tmp dir already exists, merge with data from original dir using rsync
            logger.warning(
                "-> Both original and tmp work dirs exist. Merging using rsync, where original dir takes precedence."
            )
            os.system(f"rsync --remove-source-files -avh {work_dir_orig.absolute()}/ {work_dir_tmp.absolute()}/")
            os.system(f"find {work_dir_orig.absolute()} -type d -empty -delete")  # remove empty dirs
        else:
            # If tmp dir doesn't exist, move original dir to tmp
            logger.info(f"-> Moving existing {stage.name} work dir to tmp dir...")
            shutil.move(str(work_dir_orig), str(work_dir_tmp))
    else:
        # Else, just create tmp dir
        work_dir_tmp.mkdir(parents=True, exist_ok=True)

    # Create symlink from original work dir to tmp dir
    os.symlink(work_dir_tmp, work_dir_orig)
    logger.info(f"-> {stage.name} symlinked: {work_dir_orig} -> {work_dir_tmp}.")
    return work_dir_tmp


def _move_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def teardown_tmp_work_dir(
    s: Simulation,
    stage: Stage,
    move_globs: List[str] | None = None,
) -> None:
    """Tear down temporary work dir of stage, moving results back to original location.
    If `move_globs` is provided, only move files matching these glob patterns.

    Parameters
    ----------
    s : Simulation
        Simulation object.
    stage : Stage
        Stage for which to tear down temporary work dir.
    move_globs : List[str] | None, optional
        List of glob patterns to move from tmp dir to original dir.
    """

    work_dir_orig = stage.get_work_dir(s, create=False)  # don't create or moving fails!

    # If original dir is not a symlink, something is wrong -> save output and abort.
    # Note: this check must come BEFORE `os.readlink`, which raises OSError on a non-symlink.
    if not work_dir_orig.is_symlink():
        raise RuntimeError(f"Expected {work_dir_orig} to be a symlink to a tmp dir, but it is not!")

    work_dir_tmp = pathlib.Path(os.readlink(work_dir_orig))

    # Tmp dir vanished under us (wiped scratch) -> nothing to move back, just drop the broken link
    if not work_dir_tmp.exists():
        logger.warning(f"Tmp work dir '{work_dir_tmp}' is gone. Removing dangling symlink '{work_dir_orig}'.")
        work_dir_orig.unlink()
        return

    # Remove symlink and move dir back
    if move_globs is None:
        logger.info(f"-> Moving {stage.name} results back to sim dir...")
        work_dir_orig.unlink()  # remove symlink
        shutil.move(str(work_dir_tmp), str(work_dir_orig))  # move entire dir, which removes it from tmp
    else:
        logger.info(f"-> Moving {stage.name} results back to sim dir (filtered)...")
        work_dir_orig.unlink()  # remove symlink

        for pattern in move_globs:
            logger.info(f"... using pattern '{pattern}'")

            # Create path pairs for moving: src in tmp dir, dest in orig dir
            src_dst_pairs = ((f, work_dir_orig / f.relative_to(work_dir_tmp)) for f in work_dir_tmp.glob(pattern))

            # If multiple CPUs available, move in parallel, else sequentially
            if stage.resources and stage.resources.cpus_total > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=stage.resources.cpus_per_task) as ex:
                    futures = [ex.submit(_move_file, src, dst) for src, dst in src_dst_pairs]
                    for future in concurrent.futures.as_completed(futures):
                        # Raise exceptions if any
                        future.result()
            else:
                for src, dst in src_dst_pairs:
                    _move_file(src, dst)

        # Remove tmp dir if empty
        if not any(work_dir_tmp.iterdir()):
            work_dir_tmp.rmdir()
        else:
            logger.warning(f"Tmp work dir {work_dir_tmp} not empty after moving filtered results.")

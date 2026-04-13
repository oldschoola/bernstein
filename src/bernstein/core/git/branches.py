"""Branch-based isolation: simple directory structure under .sdd/branches/.

Alternative to git worktrees for task isolation. Creates a shared working
directory per task group (parent + subtasks) instead of per-session worktrees.

Key differences from worktrees:
- Directory: `.sdd/branches/{task_id}/` (not `.sdd/worktrees/{session_id}/`)
- Sharing: Subtasks share parent's branch directory
- Git: Creates git branch but no worktree (changes tracked in main checkout)
- Simpler: No git worktree overhead, better Windows compatibility
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bernstein.core.git.git_hygiene import rmtree_windows_safe

if TYPE_CHECKING:
    from bernstein.core.git.worktree import WorktreeSetupConfig

logger = logging.getLogger(__name__)

_BRANCHES_BASE = ".sdd/branches"


class BranchError(Exception):
    """Error creating or managing a branch directory."""


@dataclass
class BranchInfo:
    """Information about a branch directory."""

    task_id: str
    path: Path
    branch_name: str
    session_count: int = 0


class BranchManager:
    """Manage per-task branch directories for agent isolation.

    Unlike WorktreeManager which creates a separate git worktree per session,
    BranchManager creates a shared directory per task group. All subtasks of
    a parent task work in the same directory.

    This is simpler and more Windows-friendly than git worktrees, at the cost
    of less isolation between concurrent subtasks.

    Args:
        repo_root: Absolute path to the repository root.
        setup_config: Optional environment setup applied after directory creation.
    """

    def __init__(
        self,
        repo_root: Path,
        setup_config: "WorktreeSetupConfig | None" = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self._base_dir = self.repo_root / _BRANCHES_BASE
        self._setup_config = setup_config
        self._shutdown_event: threading.Event | None = None
        self._branches: dict[str, BranchInfo] = {}
        self._lock = threading.Lock()

    def set_shutdown_event(self, shutdown_event: threading.Event | None) -> None:
        """Attach a shutdown event used to reject new branch creation."""
        self._shutdown_event = shutdown_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        task_id: str,
        session_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> Path:
        """Create or reuse a branch directory for a task.

        If the directory already exists (e.g., subtask of an active parent),
        it's reused. Otherwise, a new directory is created.

        Args:
            task_id: Task identifier (e.g., 'gh-59' or a UUID).
            session_id: Optional session ID for tracking (not used for path).
            parent_task_id: Optional parent task ID from Task.parent_task_id.
                If provided, subtasks share their parent's directory.
                If None, each task gets its own directory.

        Returns:
            Path to the branch directory.

        Raises:
            BranchError: If directory creation fails.
        """
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            raise BranchError("Orchestrator shutting down — refusing new branch directory")

        # Use parent_task_id for grouping if provided, otherwise use task_id
        # This handles UUID-based task IDs correctly (no regex pattern matching)
        parent_id = parent_task_id or task_id
        branch_path = self._base_dir / parent_id

        with self._lock:
            # Check if we already have this branch
            if parent_id in self._branches:
                info = self._branches[parent_id]
                info.session_count += 1
                logger.info("Reusing branch directory %s (sessions: %d)", branch_path, info.session_count)
                return info.path

            # Create new branch directory
            if branch_path.exists():
                # Directory exists from prior run - reuse it
                logger.info("Reusing existing branch directory %s", branch_path)
            else:
                self._base_dir.mkdir(parents=True, exist_ok=True)
                branch_path.mkdir(parents=True, exist_ok=True)
                logger.info("Created branch directory %s", branch_path)

                # Apply setup config if provided
                if self._setup_config is not None:
                    self._apply_setup(branch_path)

            branch_name = f"task/{parent_id}"
            info = BranchInfo(
                task_id=parent_id,
                path=branch_path,
                branch_name=branch_name,
                session_count=1,
            )
            self._branches[parent_id] = info

            return branch_path

    def release(self, task_id: str) -> bool:
        """Release a session's hold on a branch directory.

        Decrements the session count. When count reaches zero, the directory
        can be cleaned up.

        Args:
            task_id: Task identifier.

        Returns:
            True if the branch was released, False if not found.
        """
        parent_id = self._extract_parent_id(task_id)

        with self._lock:
            if parent_id not in self._branches:
                return False

            info = self._branches[parent_id]
            info.session_count -= 1
            logger.debug("Released branch %s (sessions remaining: %d)", parent_id, info.session_count)
            return True

    def cleanup(self, task_id: str, force: bool = False) -> None:
        """Remove the branch directory for a task.

        Only removes if no sessions are using it (or force=True).

        Args:
            task_id: Task identifier.
            force: Remove even if sessions are active.
        """
        parent_id = self._extract_parent_id(task_id)
        branch_path = self._base_dir / parent_id

        with self._lock:
            if parent_id in self._branches:
                info = self._branches[parent_id]
                if info.session_count > 0 and not force:
                    logger.warning(
                        "Cannot cleanup branch %s: %d sessions still active",
                        parent_id,
                        info.session_count,
                    )
                    return
                del self._branches[parent_id]

        if branch_path.exists():
            if rmtree_windows_safe(branch_path):
                logger.info("Cleaned up branch directory %s", branch_path)
            else:
                logger.warning("Failed to remove branch directory %s (file may be locked)", branch_path)

    def cleanup_all_stale(self) -> int:
        """Remove all stale branch directories from prior runs.

        Returns:
            Number of directories cleaned up.
        """
        if not self._base_dir.exists():
            return 0

        cleaned = 0
        with self._lock:
            active_ids = set(self._branches.keys())

        for entry in self._base_dir.iterdir():
            if entry.is_dir() and entry.name not in active_ids:
                if rmtree_windows_safe(entry):
                    logger.info("Cleaned stale branch directory: %s", entry.name)
                    cleaned += 1
                else:
                    logger.warning("Failed to clean stale branch %s (file may be locked)", entry.name)

        return cleaned

    def list_active(self) -> list[str]:
        """Return task IDs with active branch directories."""
        with self._lock:
            return list(self._branches.keys())

    def get_path(self, task_id: str) -> Path | None:
        """Get the branch directory path for a task, if it exists."""
        parent_id = self._extract_parent_id(task_id)
        with self._lock:
            info = self._branches.get(parent_id)
            return info.path if info else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_parent_id(self, task_id: str) -> str:
        """Extract parent task ID from a task ID.

        Examples:
            'gh-59' -> 'gh-59'
            'gh-59-sub1' -> 'gh-59'
            'gh-59-sub1-something' -> 'gh-59'
        """
        # Look for -sub pattern and strip it
        import re

        match = re.match(r"^([\w-]+?)(-sub\d+.*)?$", task_id)
        if match and match.group(2):
            return match.group(1)
        return task_id

    def _apply_setup(self, branch_path: Path) -> None:
        """Apply setup config to newly created branch directory."""
        if self._setup_config is None:
            return

        # Copy files from repo root
        for file_pattern in self._setup_config.copy_files:
            for src_file in self.repo_root.glob(file_pattern):
                if src_file.is_file():
                    dst_file = branch_path / src_file.relative_to(self.repo_root)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    logger.debug("Copied %s to branch", src_file.name)

        # Symlink directories (if supported)
        for dir_pattern in self._setup_config.symlink_dirs:
            for src_dir in self.repo_root.glob(dir_pattern):
                if src_dir.is_dir():
                    dst_dir = branch_path / src_dir.relative_to(self.repo_root)
                    if not dst_dir.exists():
                        try:
                            dst_dir.parent.mkdir(parents=True, exist_ok=True)
                            os.symlink(src_dir, dst_dir, target_is_directory=True)
                            logger.debug("Symlinked %s in branch", src_dir.name)
                        except OSError:
                            # Symlinks may not work on Windows without admin
                            logger.debug("Symlink failed for %s, skipping", src_dir.name)

        # Run setup command if provided
        if self._setup_config.setup_command:
            import subprocess

            try:
                subprocess.run(
                    self._setup_config.setup_command,
                    shell=True,
                    cwd=branch_path,
                    timeout=300,
                    check=False,
                )
            except Exception as exc:
                logger.warning("Setup command failed: %s", exc)

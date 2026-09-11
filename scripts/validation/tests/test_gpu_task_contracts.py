"""Tests for the GPU tasks that have to escape the Slurm login GPU cgroup.

The Slurm login setup puts a DeviceAllow guard on the ssh service, so anything
Ansible runs over its ssh connection sees no GPUs. Every privileged GPU command
therefore runs in a transient systemd unit. These tests pin that contract and
exercise the MIG capability probe against a mocked nvidia-smi, so the difference
between "this GPU does not support MIG" and "the query never ran" is covered.

Run with: python3 -m unittest discover scripts/validation/tests
"""

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
TRANSIENT_SERVICE = "systemd-run --wait --pipe --collect --quiet --"
PROBE_NAME = "check for MIG capable devices"
MIG_PLAYBOOK = "playbooks/nvidia-software/nvidia-mig.yml"
MIG_ROLE = "roles/nvidia-mig-manager/tasks/main.yml"
GPU_TASK_PATHS = (
    "playbooks/nvidia-software/nvidia-driver.yml",
    "playbooks/nvidia-software/nvidia-cuda.yml",
    "playbooks/utilities/gpu-clocks.yml",
    "playbooks/utilities/nvidia-set-gpu-clocks.yml",
    MIG_PLAYBOOK,
    MIG_ROLE,
)


def tasks(path):
    """Every command/shell task in a playbook or task file, flattened."""
    found = []

    def visit(value):
        if isinstance(value, dict):
            if "command" in value or "shell" in value:
                found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
    return found


def gpu_commands(path):
    """The (task, normalized command) pairs that drive the GPUs."""
    pairs = []
    for task in tasks(path):
        command = task.get("command", task.get("shell"))
        if isinstance(command, str) and (
            "nvidia-smi" in command
            or "nvidia-mig-parted apply" in command
            or "nvidia-mig-parted assert" in command
        ):
            pairs.append((task, " ".join(command.split())))
    return pairs


def all_tasks(path):
    """Every task in a playbook or task file, whatever module it uses."""
    found = []

    def visit(value):
        if isinstance(value, dict):
            # A task carries a name and a module; `include_role: {name: x}` does not.
            if isinstance(value.get("name"), str) and len(value) > 1:
                found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
    return found


def task_named(path, name):
    matches = [task for task in all_tasks(path) if task.get("name") == name]
    assert len(matches) == 1, f"{path}: {name}"
    return matches[0]


class TransientExecutionContract(unittest.TestCase):
    """Nothing that touches a GPU may run directly in the ssh context."""

    def test_every_gpu_command_runs_in_a_transient_unit(self):
        for path in GPU_TASK_PATHS:
            for task, command in gpu_commands(path):
                with self.subTest(path=path, task=task.get("name")):
                    self.assertIn(TRANSIENT_SERVICE, command)
                    driven = "nvidia-smi" if "nvidia-smi" in command else "nvidia-mig-parted"
                    self.assertLess(
                        command.index(TRANSIENT_SERVICE), command.index(driven)
                    )

    def test_mig_administration_is_covered_too(self):
        # The probe is useless on its own: once it succeeds, apply and assert
        # drive the very same GPUs and need the same isolation.
        commands = [command for _, command in gpu_commands(MIG_PLAYBOOK)]
        for fragment in (
            "nvidia-smi --query-gpu=mig.mode.current",
            "nvidia-mig-parted apply",
            "nvidia-mig-parted assert",
        ):
            with self.subTest(fragment=fragment):
                matching = [c for c in commands if fragment in c]
                self.assertEqual(len(matching), 1, fragment)
                self.assertIn(TRANSIENT_SERVICE, matching[0])

    def test_mig_operations_run_privileged(self):
        plays = yaml.safe_load((ROOT / MIG_PLAYBOOK).read_text(encoding="utf-8"))
        self.assertEqual(len(plays), 1)
        self.assertIs(plays[0]["become"], True)
        self.assertIs(task_named(MIG_ROLE, PROBE_NAME)["become"], True)

    def test_gpu_clock_arguments_avoid_shell_interpolation(self):
        path = "playbooks/utilities/nvidia-set-gpu-clocks.yml"
        for name in (
            "set the gpu clock to a specified amount",
            "reset the gpu clock to the default",
        ):
            with self.subTest(name=name):
                task = task_named(path, name)
                self.assertIn("command", task)
                self.assertNotIn("shell", task)


class MigApplyFailureContract(unittest.TestCase):
    """A MIG layout that could not be applied must not pass as applied."""

    def test_apply_and_assert_surface_their_failures(self):
        for name in ("Apply MIG configuration", "Assert MIG configuration was applied"):
            with self.subTest(name=name):
                task = task_named(MIG_PLAYBOOK, name)
                self.assertNotIn("failed_when", task)
                self.assertNotIn("ignore_errors", task)

    def test_installation_requires_a_successful_capability_probe(self):
        # Without this the Red Hat branch installed the MIG manager on nodes
        # whose capability was never established.
        for name, family in (
            ("Install MIG Manager (apt)", "Debian"),
            ("Install MIG Manager (yum)", "RedHat"),
        ):
            with self.subTest(name=name):
                tasks_by_name = {
                    task["name"]: task
                    for task in yaml.safe_load(
                        (ROOT / MIG_ROLE).read_text(encoding="utf-8")
                    )
                }
                self.assertEqual(
                    set(tasks_by_name[name]["when"]),
                    {
                        "has_mig.rc == 0",
                        "has_mig_parted.rc != 0",
                        f'ansible_os_family == "{family}"',
                    },
                )

    def test_a_failed_probe_is_reported_rather_than_skipped(self):
        # rc 2 is the probe's own "the query never ran"; the lspci fact keeps
        # nodes without NVIDIA hardware out of it.
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                guard = task_named(path, "fail when the MIG capability probe could not run")
                self.assertEqual(
                    set(guard["when"]),
                    {
                        "has_mig.rc == 2",
                        "ansible_local['gpus']['count'] | default(0) | int > 0",
                    },
                )


class MigProbeBehaviour(unittest.TestCase):
    """Run the probe as written, against a mocked nvidia-smi."""

    def probe(self, path):
        probe = task_named(path, PROBE_NAME)
        self.assertIs(probe["failed_when"], False)
        self.assertIs(probe["changed_when"], False)
        return probe["shell"]

    def run_probe(self, path, mode, ssh_device_guard=False):
        """Execute the probe's shell exactly as Ansible would."""
        script = self.probe(path)
        with tempfile.TemporaryDirectory() as tmp:
            mocks = Path(tmp)

            # systemd-run drops the ssh cgroup: it consumes its own options up to
            # the '--' separator and runs the rest outside the device guard.
            (mocks / "systemd-run").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    while [ "$1" != "--" ]; do
                      case "$1" in
                        --wait|--pipe|--collect|--quiet) shift ;;
                        *) echo "systemd-run: unexpected argument $1" >&2; exit 1 ;;
                      esac
                    done
                    shift
                    SSH_DEVICE_GUARD=0 exec "$@"
                    """
                ),
                encoding="utf-8",
            )
            # nvidia-smi reports a missing device on stdout and a broken driver
            # on stderr, so the probe must not depend on which stream carried it.
            (mocks / "nvidia-smi").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [ "${SSH_DEVICE_GUARD:-0}" = "1" ]; then
                      echo "No devices were found"
                      exit 6
                    fi
                    case "$MODE" in
                      mig)      printf 'Enabled\\nDisabled\\n'; exit 0 ;;
                      no-mig)   printf 'N/A\\nN/A\\n'; exit 0 ;;
                      noisy)    printf 'N/A\\nWARNING: infoROM is corrupted\\n'; exit 0 ;;
                      hidden)   echo "No devices were found"; exit 6 ;;
                      nodriver) echo "NVIDIA-SMI has failed to communicate with the driver" >&2
                                exit 9 ;;
                      missing)  exit 127 ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            for mock in ("systemd-run", "nvidia-smi"):
                (mocks / mock).chmod(0o755)

            return subprocess.run(
                ["/bin/bash", "-c", script],
                env={
                    **os.environ,
                    "PATH": f"{mocks}:{os.environ['PATH']}",
                    "MODE": mode,
                    "SSH_DEVICE_GUARD": "1" if ssh_device_guard else "0",
                },
                text=True,
                capture_output=True,
                check=False,
            )

    def assert_operational_failure(self, result):
        """What the playbook reads as "the query never ran"."""
        self.assertEqual(result.returncode, 2)

    def assert_unsupported_device(self, result):
        """What the playbook reads as "these GPUs do not do MIG"."""
        self.assertEqual(result.returncode, 1)

    def test_mig_capable_devices_are_detected(self):
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                result = self.run_probe(path, "mig")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.split(), ["Enabled", "Disabled"])

    def test_devices_without_mig_support_skip_configuration(self):
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                self.assert_unsupported_device(self.run_probe(path, "no-mig"))

    def test_only_a_real_mig_mode_counts_as_a_capability(self):
        # nvidia-smi prints warnings alongside the values it was asked for, and
        # 'grep -v N/A' accepted those as a MIG mode.
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                self.assert_unsupported_device(self.run_probe(path, "noisy"))

    def test_a_failed_query_does_not_look_like_a_missing_capability(self):
        # Whether the diagnostic went to stdout, to stderr or nowhere at all,
        # a query that did not answer has to be distinguishable.
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            for mode in ("hidden", "nodriver", "missing"):
                with self.subTest(path=path, mode=mode):
                    self.assert_operational_failure(self.run_probe(path, mode))

    def test_the_probe_escapes_an_active_ssh_device_guard(self):
        # The regression this PR is about: with the guard already applied, the
        # bare command sees no devices, and the transient unit still does.
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                result = self.run_probe(path, "mig", ssh_device_guard=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_probe_is_stable_across_repeated_runs(self):
        # Convergence runs after the login isolation is in place have to reach
        # the same answer as the run that applied it.
        for path in (MIG_PLAYBOOK, MIG_ROLE):
            with self.subTest(path=path):
                first = self.run_probe(path, "mig", ssh_device_guard=False)
                second = self.run_probe(path, "mig", ssh_device_guard=True)
                self.assertEqual(first.returncode, second.returncode)
                self.assertEqual(first.stdout, second.stdout)


if __name__ == "__main__":
    unittest.main()

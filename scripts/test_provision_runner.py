# SPDX-License-Identifier: Apache-2.0

"""Tests for the sizing half of scripts/provision_runner.py: reading a node's layout and choosing
the container's CPUs. These are pure functions over the text a node reports, so they run anywhere.

    python -m pytest scripts/test_provision_runner.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provision_runner as pr  # noqa: E402


def two_socket_probe(cores_per_socket: int) -> str:
    """Probe output for a 2-socket hyperthreaded node numbered the way Linux numbers a Dell
    R730: node 0 holds the even CPUs, node 1 the odd ones, and CPU n's sibling is n + (all cores)."""
    cores = 2 * cores_per_socket
    lines = [
        "node 0 " + ",".join(str(cpu) for cpu in range(0, 2 * cores, 2)),
        "node 1 " + ",".join(str(cpu) for cpu in range(1, 2 * cores, 2)),
    ]
    for cpu in range(2 * cores):
        first = cpu % cores
        lines.append(f"cpu {cpu} {first},{first + cores}")
    lines += ["memavailable 154533888", "cgroup cgroup2fs"]
    return "\n".join(lines) + "\n"


def test_parse_cpulist_expands_ranges_and_singles():
    assert pr.parse_cpulist("0-3,8,12-13\n") == [0, 1, 2, 3, 8, 12, 13]


def test_format_cpulist_compacts_runs():
    assert pr.format_cpulist([72, 28, 29, 30, 40]) == "28-30,40,72"


def test_format_cpulist_round_trips_through_parse():
    cpus = [1, 3, 4, 5, 9, 10, 44]
    assert pr.parse_cpulist(pr.format_cpulist(cpus)) == cpus


def test_parse_topology_reads_nodes_siblings_memory_and_cgroup():
    topology = pr.parse_topology(two_socket_probe(2))
    assert topology.nodes == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
    assert topology.siblings[0] == (0, 4)
    assert topology.siblings[5] == (1, 5)
    assert topology.mem_available_mib == 150912
    assert topology.cgroup == "cgroup2fs"


def test_parse_topology_leaves_out_a_node_without_cpus():
    probe = "node 0 0-1\nnode 1 \ncpu 0 0\ncpu 1 1\nmemavailable 2048\ncgroup cgroup2fs\n"
    assert pr.parse_topology(probe).nodes == {0: [0, 1]}


def test_parse_topology_refuses_output_without_memory():
    with pytest.raises(pr.ProvisionError, match="could not read"):
        pr.parse_topology("node 0 0-1\ncpu 0 0\ncpu 1 1\ncgroup cgroup2fs\n")


def test_parse_topology_refuses_a_cpu_without_sibling_data():
    with pytest.raises(pr.ProvisionError, match="could not read"):
        pr.parse_topology("node 0 0-1\ncpu 0 0\nmemavailable 2048\ncgroup cgroup2fs\n")


def test_select_cpus_takes_whole_cores_evenly_from_the_top_of_each_node():
    topology = pr.parse_topology(two_socket_probe(22))
    cpus = pr.select_cpus(topology, 32)
    assert pr.format_cpulist(cpus) == "28-43,72-87"
    for index, node_cpus in topology.nodes.items():
        assert len(set(cpus) & set(node_cpus)) == 16, index
    assert all(set(topology.siblings[cpu]) <= set(cpus) for cpu in cpus)


def test_select_cpus_can_take_the_whole_node():
    topology = pr.parse_topology(two_socket_probe(22))
    assert pr.select_cpus(topology, 88) == list(range(88))


def test_select_cpus_refuses_a_count_that_splits_a_core_or_a_node():
    topology = pr.parse_topology(two_socket_probe(22))
    with pytest.raises(pr.ProvisionError, match="multiple of 4"):
        pr.select_cpus(topology, 30)


def test_select_cpus_refuses_more_than_the_node_has():
    topology = pr.parse_topology(two_socket_probe(22))
    with pytest.raises(pr.ProvisionError, match="can give is 88"):
        pr.select_cpus(topology, 92)


def test_select_cpus_refuses_zero():
    topology = pr.parse_topology(two_socket_probe(22))
    with pytest.raises(pr.ProvisionError, match="positive multiple"):
        pr.select_cpus(topology, 0)


def test_select_cpus_on_one_node_without_hyperthreads():
    probe = "node 0 0-3\n" + "".join(f"cpu {cpu} {cpu}\n" for cpu in range(4))
    probe += "memavailable 8388608\ncgroup cgroup2fs\n"
    assert pr.select_cpus(pr.parse_topology(probe), 2) == [2, 3]


def test_check_arguments_refuses_a_static_ip_without_a_gateway():
    args = pr.build_parser().parse_args(
        ["--host", "root@node", "--cores", "4", "--memory", "4096", "--ip", "10.0.0.5/24"])
    with pytest.raises(pr.ProvisionError, match="--gateway"):
        pr.check_arguments(args)


def test_check_arguments_refuses_a_malformed_mac():
    args = pr.build_parser().parse_args(
        ["--host", "root@node", "--cores", "4", "--memory", "4096", "--hwaddr", "BC:24:11"])
    with pytest.raises(pr.ProvisionError, match="not a MAC address"):
        pr.check_arguments(args)


def test_cores_and_memory_are_required():
    with pytest.raises(SystemExit):
        pr.build_parser().parse_args(["--host", "root@node"])


def test_login_file_can_leave_the_password_out(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "SCRIPT_DIR", str(tmp_path))
    login = pr.Login(ctid=250, hostname="box", node="root@node", address="10.0.0.5",
                     mdns_name="box", user="tetradrome", password="s3cret-value",
                     key_path="/keys/ct250-ssh-key", repo="/opt/t/src", python="/opt/t/venv/bin/python")
    with_password = open(pr.write_login_file(login)).read()
    without_password = open(pr.write_login_file(login, include_password=False)).read()
    assert "password: s3cret-value" in with_password.splitlines()
    assert "s3cret-value" not in without_password
    assert "address:  10.0.0.5" in without_password.splitlines()
    assert "user:     tetradrome" in without_password.splitlines()

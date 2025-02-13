import os
import time
from pathlib import Path

import pytest
import requests

# Constants
BASE_URL = "http://localhost:8000"
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "simulation_config.yaml"


@pytest.fixture
def simulation_instance():
    """Fixture that provides a running simulation instance for testing.

    Returns:
        str: The ID of the created simulation instance.
    """
    simulation_config = {
        "config_file": str(TEST_CONFIG_PATH.resolve()),
        "auto_run": True,
    }
    response = requests.post(f"{BASE_URL}/start_simulation", json=simulation_config)
    assert response.status_code == 200
    instance_id = response.json()["simulation_id"]
    yield instance_id

    # Cleanup: Stop the simulation after the test
    requests.post(
        f"{BASE_URL}/simulation_control/{instance_id}", json={"command": "stop"}
    )


def test_should_start_simulation_successfully():
    simulation_config = {
        "config_file": "/home/haoweis/terasim_vru_dev/TeraSim-Service/simulation_config.yaml",
        "auto_run": True,
    }
    response = requests.post(f"{BASE_URL}/start_simulation", json=simulation_config)
    assert response.status_code == 200
    assert "simulation_id" in response.json()


def test_should_return_valid_simulation_status(simulation_instance):
    response = requests.get(f"{BASE_URL}/simulation_status/{simulation_instance}")
    assert response.status_code == 200
    assert "status" in response.json()


def test_should_pause_and_resume_simulation_successfully(simulation_instance):
    # Pause the simulation
    pause_response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_instance}",
        json={"command": "pause"},
    )
    assert pause_response.status_code == 200

    # Resume the simulation
    resume_response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_instance}",
        json={"command": "resume"},
    )
    assert resume_response.status_code == 200


def test_should_stop_simulation_successfully(simulation_instance):
    response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_instance}", json={"command": "stop"}
    )
    assert response.status_code == 200


def test_should_execute_single_simulation_tick(simulation_instance):
    # First, pause the simulation to ensure it's not in auto-run mode
    requests.post(
        f"{BASE_URL}/simulation_control/{simulation_instance}",
        json={"command": "pause"},
    )

    tick_response = requests.post(f"{BASE_URL}/simulation_tick/{simulation_instance}")
    assert tick_response.status_code == 200


@pytest.mark.parametrize("auto_run_enabled", [True, False])
def test_should_handle_different_auto_run_modes(auto_run_enabled):
    simulation_config = {
        "config_file": "/home/haoweis/terasim_vru_dev/TeraSim-Service/simulation_config.yaml",
        "auto_run": auto_run_enabled,
    }
    start_response = requests.post(
        f"{BASE_URL}/start_simulation", json=simulation_config
    )
    assert start_response.status_code == 200
    instance_id = start_response.json()["simulation_id"]

    # Wait for a short time to allow the simulation to start
    time.sleep(2)

    # Check status
    status_response = requests.get(f"{BASE_URL}/simulation_status/{instance_id}")
    assert status_response.status_code == 200

    # Clean up
    requests.post(
        f"{BASE_URL}/simulation_control/{instance_id}", json={"command": "stop"}
    )

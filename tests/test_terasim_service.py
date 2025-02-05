import pytest
import requests
import time

BASE_URL = "http://localhost:8000"


@pytest.fixture
def simulation_id():
    # Start a simulation and return the simulation_id
    config = {
        "config_file": "/home/haoweis/terasim_vru_dev/TeraSim-Service/simulation_config.yaml",
        "auto_run": True,
    }
    response = requests.post(f"{BASE_URL}/start_simulation", json=config)
    assert response.status_code == 200
    simulation_id = response.json()["simulation_id"]
    yield simulation_id

    # Cleanup: Stop the simulation after the test
    requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "stop"}
    )


def test_start_simulation():
    config = {
        "config_file": "/home/haoweis/terasim_vru_dev/TeraSim-Service/simulation_config.yaml",
        "auto_run": True,
    }
    response = requests.post(f"{BASE_URL}/start_simulation", json=config)
    assert response.status_code == 200
    assert "simulation_id" in response.json()


def test_check_simulation_status(simulation_id):
    response = requests.get(f"{BASE_URL}/simulation_status/{simulation_id}")
    assert response.status_code == 200
    assert "status" in response.json()


def test_pause_resume_simulation(simulation_id):
    # Pause the simulation
    response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "pause"}
    )
    assert response.status_code == 200

    # Resume the simulation
    response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "resume"}
    )
    assert response.status_code == 200


def test_stop_simulation(simulation_id):
    response = requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "stop"}
    )
    assert response.status_code == 200


def test_tick_simulation(simulation_id):
    # First, pause the simulation to ensure it's not in auto-run mode
    requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "pause"}
    )

    response = requests.post(f"{BASE_URL}/simulation_tick/{simulation_id}")
    assert response.status_code == 200


@pytest.mark.parametrize("auto_run", [True, False])
def test_simulation_with_auto_run(auto_run):
    config = {
        "config_file": "/home/haoweis/terasim_vru_dev/TeraSim-Service/simulation_config.yaml",
        "auto_run": auto_run,
    }
    response = requests.post(f"{BASE_URL}/start_simulation", json=config)
    assert response.status_code == 200
    simulation_id = response.json()["simulation_id"]

    # Wait for a short time to allow the simulation to start
    time.sleep(2)

    # Check status
    status_response = requests.get(f"{BASE_URL}/simulation_status/{simulation_id}")
    assert status_response.status_code == 200

    # Clean up
    requests.post(
        f"{BASE_URL}/simulation_control/{simulation_id}", json={"command": "stop"}
    )

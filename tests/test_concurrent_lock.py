import threading
import time
import logging
import pytest
from src.database_manager import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def test_concurrent_advisory_lock_mocked(monkeypatch):
    """
    Mock the database engine's try_advisory_lock to return True for the first call
    and False for the second concurrent call, proving the context manager handles it.
    """
    db = get_db()

    class MockConnection:
        def __init__(self):
            self.lock_held = False

        def execute(self, query, params=None):
            class MockScalar:
                def __init__(self, val):
                    self.val = val
                def scalar(self):
                    return self.val

            query_str = str(query)
            if "pg_try_advisory_lock" in query_str:
                if not self.lock_held:
                    self.lock_held = True
                    return MockScalar(True)
                else:
                    return MockScalar(False)
            elif "pg_advisory_unlock" in query_str:
                self.lock_held = False
                return MockScalar(True)
            return MockScalar(None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_conn = MockConnection()

    def mock_connect():
        return mock_conn

    monkeypatch.setattr(db.engine, "connect", mock_connect)

    results = []

    def task1():
        with db.advisory_lock(12345) as acquired:
            if acquired:
                results.append("Task1 Acquired")
                time.sleep(0.5)
            else:
                results.append("Task1 Failed")

    def task2():
        with db.advisory_lock(12345) as acquired:
            if acquired:
                results.append("Task2 Acquired")
            else:
                results.append("Task2 Failed")

    t1 = threading.Thread(target=task1)
    t2 = threading.Thread(target=task2)

    t1.start()
    time.sleep(0.1)
    t2.start()

    t1.join()
    t2.join()

    assert "Task1 Acquired" in results
    assert "Task2 Failed" in results

    print("\nConcurrency Lock Test PASSED: Task 1 acquired, Task 2 was blocked and failed immediately.")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

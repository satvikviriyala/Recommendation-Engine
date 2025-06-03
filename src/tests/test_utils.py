import unittest
import os
import yaml
import logging
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Create config file in the same directory as this test script
        self.test_config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
        self.test_config_content = {
            "key1": "value1",
            "key2": "Hello, ${TEST_ENV_VAR}!",
            "nested": {
                "key3": "value3"
            }
        }
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config_content, f)

        os.environ["TEST_ENV_VAR"] = "World"
        # For testing default config path
        self.default_config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
        self.default_config_path = os.path.join(self.default_config_dir, "config.yaml")
        os.makedirs(self.default_config_dir, exist_ok=True)
        with open(self.default_config_path, 'w') as f:
            yaml.dump({"default_key": "default_value"}, f)


    def tearDown(self):
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if "TEST_ENV_VAR" in os.environ:
            del os.environ["TEST_ENV_VAR"]
        if os.path.exists(self.default_config_path):
            os.remove(self.default_config_path)
        if os.path.exists(self.default_config_dir) and not os.listdir(self.default_config_dir):
            os.rmdir(self.default_config_dir)


    def test_load_config_loads_specific_file(self):
        config = load_config(self.test_config_path)
        self.assertIsNotNone(config)
        self.assertEqual(config.get("key1"), "value1")
        self.assertEqual(config.get("nested", {}).get("key3"), "value3")

    def test_load_config_substitutes_env_vars(self):
        config = load_config(self.test_config_path)
        self.assertEqual(config.get("key2"), "Hello, World!")

    def test_load_config_default_path(self):
        # Test that it can load from configs/config.yaml relative to src/utils if no path given
        # This requires load_config to be aware of its own location or project root
        # For this test, we assume load_config() without args tries to load 'configs/config.yaml'
        # from project root. The setUp created this dummy file.
        # Note: This test depends on the CWD or how load_config resolves the default path.
        # If load_config uses __file__ to find project root, this test might need adjustment
        # based on where the tests are run from.
        # For now, assuming load_config() will find the dummy configs/config.yaml
        original_cwd = os.getcwd()
        # Assuming project root is two levels up from src/tests
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(project_root) # Change CWD to project root for load_config default path
        try:
            config = load_config() # Test default path
            self.assertIsNotNone(config)
            self.assertEqual(config.get("default_key"), "default_value")
        finally:
            os.chdir(original_cwd) # Change back to original CWD

    def test_load_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config.yaml")

class TestLoggingUtils(unittest.TestCase):
    def test_setup_logger_returns_logger(self):
        logger = setup_logger("test_logger_instance", logging.DEBUG)
        self.assertIsInstance(logger, logging.Logger)
        # Clean up
        if logger.hasHandlers():
            logger.handlers.clear()

    def test_setup_logger_sets_level(self):
        logger = setup_logger("test_logger_level", logging.WARNING)
        self.assertEqual(logger.level, logging.WARNING)
        # Clean up
        if logger.hasHandlers():
            logger.handlers.clear()

    def test_setup_logger_adds_handler(self):
        logger = setup_logger("test_logger_handler", logging.INFO)
        self.assertTrue(logger.hasHandlers())
        # Clean up
        if logger.hasHandlers():
            logger.handlers.clear()

    def test_setup_logger_singleton_behavior(self):
        # Get logger first time
        logger1 = setup_logger("singleton_test_logger", logging.INFO)
        initial_handler_count = len(logger1.handlers)
        self.assertGreaterEqual(initial_handler_count, 1, "Logger should have at least one handler initially.")

        # Get logger second time with same name
        logger2 = setup_logger("singleton_test_logger", logging.INFO)

        self.assertIs(logger1, logger2, "setup_logger should return the same logger instance for the same name.")
        self.assertEqual(len(logger2.handlers), initial_handler_count, "setup_logger should not add duplicate handlers to an existing logger.")

        # Clean up handlers for this specific logger to avoid interference
        # Get the actual logger instance (it might be a proxy or wrapped)
        actual_logger = logging.getLogger("singleton_test_logger")
        actual_logger.handlers.clear()


if __name__ == '__main__':
    unittest.main()

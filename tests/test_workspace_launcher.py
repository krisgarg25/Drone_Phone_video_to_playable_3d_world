import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


class WorkspaceLauncherTests(unittest.TestCase):
    def test_existing_services_open_next_workspace_without_new_processes(self):
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.return_value = 0
        with patch.object(pipeline.socket, "socket", return_value=connection), \
             patch.object(pipeline.webbrowser, "open") as browser, \
             patch.object(pipeline.subprocess, "Popen") as spawn:
            pipeline.do_ui()
        browser.assert_called_once_with("http://127.0.0.1:3000")
        spawn.assert_not_called()

    def test_missing_dependencies_fail_without_starting_services(self):
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.return_value = 1
        with patch.object(pipeline.socket, "socket", return_value=connection), \
             patch.object(pipeline.shutil, "which", return_value=None), \
             patch.object(pipeline.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(SystemExit, "Node.js"):
                pipeline.do_ui()
        spawn.assert_not_called()

    def test_new_services_are_local_and_only_owned_children_are_stopped(self):
        import _serve
        ports = set()
        children = []
        commands = []
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.side_effect = lambda address: 0 if address[1] in ports else 1
        def start(command, **options):
            commands.append(command)
            ports.add(3000 if "dev" in command else 8137)
            child = MagicMock()
            child.poll.return_value = None
            children.append(child)
            return child
        with patch.object(pipeline.socket, "socket", return_value=connection), \
             patch.object(pipeline.shutil, "which", return_value="node"), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(pipeline.subprocess, "Popen", side_effect=start), \
             patch.object(pipeline.time, "sleep", side_effect=KeyboardInterrupt), \
             patch.object(pipeline.webbrowser, "open") as browser, \
             patch.object(_serve, "terminate_process_tree") as stop:
            pipeline.do_ui()
        self.assertEqual(len(commands), 2)
        self.assertIn("127.0.0.1", commands[0][2])
        self.assertEqual(commands[1][-4:], ["--hostname", "127.0.0.1", "--port", "3000"])
        browser.assert_called_once_with("http://127.0.0.1:3000")
        self.assertEqual([call.args[0] for call in stop.call_args_list], list(reversed(children)))

    def test_view_uses_selected_next_project(self):
        with patch.object(pipeline, "do_ui") as launch, \
             patch.object(pipeline.webbrowser, "open"), \
             patch.object(pipeline.subprocess, "run"), \
             patch.object(Path, "exists", return_value=True):
            pipeline.do_view({"name": "rocks", "work": pipeline.ROOT / "work/rocks"})
        launch.assert_called_once_with("rocks")


if __name__ == "__main__":
    unittest.main()

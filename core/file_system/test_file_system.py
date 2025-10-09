import pytest
from pathlib import Path
import json

from core.file_system.file_system import FileSystem

@pytest.fixture
def temp_workspace(tmp_path):
    """Provides a temporary workspace directory for each test."""
    return tmp_path / "workspace"

@pytest.fixture
def file_system(temp_workspace):
    """Provides a fresh FileSystem instance for each test."""
    return FileSystem(temp_workspace)

class TestFileSystem:

    def test_write_and_read_file(self, file_system):
        """Test writing to and reading from a file."""
        path = "test_file.txt"
        content = "Hello, World!"
        
        write_status = file_system.writeFile(path, content)
        assert "Successfully wrote" in write_status
        
        read_content = file_system.readFile(path)
        assert read_content == content

    def test_read_non_existent_file(self, file_system):
        """Test reading a file that does not exist."""
        path = "non_existent_file.txt"
        read_result = file_system.readFile(path)
        error_info = json.loads(read_result)
        assert error_info["error"]["type"] == "FileNotFound"

    def test_path_security(self, file_system):
        """Test that the file system prevents directory traversal."""
        with pytest.raises(ValueError, match="Security Error"):
            file_system.writeFile("../outside_file.txt", "test")
            
        with pytest.raises(ValueError, match="Security Error"):
            file_system.readFile("../outside_file.txt")

    def test_write_to_subdirectory(self, file_system):
        """Test writing to a file in a subdirectory that doesn't exist yet."""
        path = "subdir/test_file.txt"
        content = "Subdir content"
        
        write_status = file_system.writeFile(path, content)
        assert "Successfully wrote" in write_status
        
        read_content = file_system.readFile(path)
        assert read_content == content

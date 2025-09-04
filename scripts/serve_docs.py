import subprocess
import sys
from pathlib import Path


def main():
    """Serve documentation locally."""
    root_dir = Path(__file__).parent.parent
    docs_dir = root_dir / "agent-sdk-reference"

    if not docs_dir.exists():
        print("Documentation not found. Run 'build_docs' first.")
        return

    print("Serving documentation at http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run([
            sys.executable, "-m", "http.server", "8000"
        ], cwd=docs_dir, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()


import sys
import subprocess
import shutil
from pathlib import Path


def build_docs_for_path(path, output_dir, name):
    """Build documentation for a specific path, handling import errors gracefully."""
    try:
        print(f"Building documentation for {name}...")

        if output_dir.exists():
            shutil.rmtree(output_dir)

        result = subprocess.run([
            sys.executable, "-m", "pdoc",
            "--html",
            "--skip-errors",
            "--output-dir", str(output_dir),
            str(path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Successfully built documentation for {name}")
            return True
        else:
            print(f"✗ Failed to build documentation for {name}")
            if result.stderr:
                error_lines = result.stderr.split('\n')
                for line in error_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
            return False

    except Exception as e:
        print(f"✗ Error building documentation for {name}: {e}")
        return False


def main():
    """Build documentation for all packages."""
    root_dir = Path(__file__).parent.parent

    try:
        import pdoc
    except ImportError:
        print("Installing pdoc3...")
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "pdoc3"], check=True)

    docs_dir = root_dir / "docs" / "api"
    docs_dir.mkdir(parents=True, exist_ok=True)

    agents_path = root_dir / "videosdk-agents" / "videosdk" / "agents"
    if agents_path.exists():
        if build_docs_for_path(agents_path, docs_dir / "agents", "videosdk-agents"):
            print(f"✓ Successfully built documentation for videosdk-agents")

    plugins_dir = root_dir / "videosdk-plugins"
    if plugins_dir.exists():
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir() and plugin_dir.name.startswith("videosdk-plugins-"):
                plugin_name = plugin_dir.name.replace("videosdk-plugins-", "")
                plugin_path = plugin_dir / "videosdk" / "plugins" / plugin_name

                if plugin_path.exists():
                    if build_docs_for_path(plugin_path, docs_dir / f"plugins-{plugin_name}", plugin_name):
                        print(
                            f"✓ Successfully built documentation for {plugin_name}")

    print(f"\nDocumentation build completed!")
    print(f"Output directory: {docs_dir}")


if __name__ == "__main__":
    main()


import os
import sys
import subprocess
import shutil
from pathlib import Path


def flatten_plugin_docs(plugin_output_dir, plugin_name):
    """Flatten the plugin documentation structure by moving files up from nested directories."""
    try:
        nested_path = plugin_output_dir / "videosdk" / "plugins" / plugin_name

        if nested_path.exists():
            for html_file in nested_path.glob("*.html"):
                target_file = plugin_output_dir / html_file.name
                if target_file.exists():
                    target_file.unlink()
                html_file.rename(target_file)

            shutil.rmtree(plugin_output_dir / "videosdk")
        else:
            print(
                f"Nested path not found for {plugin_name}: {nested_path}")

    except Exception as e:
        print(f"Error flattening docs for {plugin_name}: {e}")


def flatten_agents_docs(agents_output_dir):
    """Flatten the agents documentation structure by moving files up from nested directories."""
    try:
        nested_path = agents_output_dir / "agents"

        if nested_path.exists():
            for item in nested_path.iterdir():
                target_item = agents_output_dir / item.name
                if target_item.exists():
                    if target_item.is_file():
                        target_item.unlink()
                    else:
                        shutil.rmtree(target_item)
                item.rename(target_item)

            shutil.rmtree(agents_output_dir / "agents")
            print(f"Flattened agents documentation structure")
        else:
            print(f"Nested agents path not found: {nested_path}")

    except Exception as e:
        print(f"Error flattening agents docs: {e}")


def build_docs_for_path(path, output_dir, name, python_executable):
    """Build documentation for a specific path."""
    try:
        print(f"Building documentation for {name}...")

        if output_dir.exists():
            shutil.rmtree(output_dir)

        env = os.environ.copy()
        working_dir = None
        module_path = str(path)

        if "plugins" in str(path):
            plugin_root = path.parent.parent.parent
            env["PYTHONPATH"] = f"{plugin_root}:{env.get('PYTHONPATH', '')}"
            working_dir = str(plugin_root)
            module_path = f"videosdk.plugins.{name}"

        cmd = [
            python_executable, "-m", "pdoc",
            "--html",
            "--output-dir", str(output_dir),
            module_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=working_dir
        )

        if result.returncode == 0:
            if "plugins" in str(path):
                flatten_plugin_docs(output_dir, name)
            elif "agents" in str(path):
                flatten_agents_docs(output_dir)

            return True
        else:
            if result.stderr:
                for line in result.stderr.split('\n')[-5:]:
                    if line.strip():
                        print(f"  {line}")
            return False

    except Exception as e:
        print(f"Error building documentation for {name}: {e}")
        return False


def get_python_executable():
    """Get the appropriate Python executable (venv preferred)."""
    root_dir = Path(__file__).parent.parent
    venv_python = root_dir / "venv" / "bin" / "python"

    if venv_python.exists():
        python_executable = str(venv_python)
        print(f"Using virtual environment Python: {python_executable}")
    else:
        python_executable = sys.executable
        print(f"Using system Python: {python_executable}")

    return python_executable


def ensure_pdoc_installed(python_executable):
    """Ensure pdoc3 is installed."""
    try:
        import pdoc
        print("pdoc3 is already installed")
    except ImportError:
        print("Installing pdoc3...")
        subprocess.run([python_executable, "-m", "pip",
                       "install", "pdoc3"], check=True)


def build_agents_docs(root_dir, docs_dir, python_executable):
    """Build documentation for the main agents package."""
    agents_path = root_dir / "videosdk-agents" / "videosdk" / "agents"
    if agents_path.exists():
        build_docs_for_path(agents_path, docs_dir / "agents",
                            "videosdk-agents", python_executable)


def build_plugin_docs(root_dir, docs_dir, python_executable):
    """Build documentation for all plugins."""
    plugins_dir = root_dir / "videosdk-plugins"
    if not plugins_dir.exists():
        return

    for plugin_dir in plugins_dir.iterdir():
        if not (plugin_dir.is_dir() and plugin_dir.name.startswith("videosdk-plugins-")):
            continue

        plugin_name = plugin_dir.name.replace("videosdk-plugins-", "")
        plugin_path = plugin_dir / "videosdk" / "plugins" / plugin_name

        if plugin_path.exists():
            output_dir = docs_dir / f"plugins-{plugin_name}"
            success = build_docs_for_path(
                plugin_path, output_dir, plugin_name, python_executable)

            if success:
                print(f"Successfully built documentation for {plugin_name}")
            else:
                print(f"Failed to build documentation for {plugin_name}")


def main():
    """Build documentation for all packages."""
    root_dir = Path(__file__).parent.parent
    python_executable = get_python_executable()
    ensure_pdoc_installed(python_executable)

    docs_dir = root_dir / "docs" / "api"
    docs_dir.mkdir(parents=True, exist_ok=True)

    build_agents_docs(root_dir, docs_dir, python_executable)
    build_plugin_docs(root_dir, docs_dir, python_executable)

    print(f"\nDocumentation build completed!")
    print(f"Output directory: {docs_dir}")


if __name__ == "__main__":
    main()

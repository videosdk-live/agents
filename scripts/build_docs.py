
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def remove_version_files(output_dir):
    """Remove version.py files and their references from the generated documentation."""
    try:
        # Remove version.html files
        for html_file in output_dir.rglob("*.html"):
            if "version" in html_file.name.lower():
                html_file.unlink()

                # Clean up references in all HTML files
        for html_file in output_dir.rglob("*.html"):
            try:
                content = html_file.read_text(encoding='utf-8')

                # Remove version module references from the index
                # Handle both <dt> and <li> patterns
                import re

                # Remove <dt> pattern: <dt><code class="name"><a title="module.version" href="version.html">[^<]*\.version</a></code></dt>
                content = re.sub(
                    r'<dt><code class="name"><a title="[^"]*\.version" href="version\.html">[^<]*\.version</a></code></dt>\s*',
                    '',
                    content
                )

                # Remove <li> pattern: <li><code><a title="module.version" href="version.html">[^<]*\.version</a></code></li>
                content = re.sub(
                    r'<li><code><a title="[^"]*\.version" href="version\.html">[^<]*\.version</a></code></li>\s*',
                    '',
                    content
                )

                # Also remove any empty <dd> tags that might be left behind
                content = re.sub(r'<dd>\s*</dd>\s*', '', content)

                # Fix href paths to use explicit relative paths with ./
                # Pattern: href="filename.html" -> href="./filename.html"
                content = re.sub(
                    r'href="([^"]*\.html)"',
                    r'href="./\1"',
                    content
                )

                # Also fix href="index.html" -> href="./index.html" for current directory
                content = re.sub(
                    r'href="index\.html"',
                    r'href="./index.html"',
                    content
                )

                # Fix supermodule references: href="module.html" -> href="./module.html" for parent modules
                # Pattern: href="agents.html" -> href="./agents.html" (when in submodule)
                content = re.sub(
                    r'href="([a-zA-Z_][a-zA-Z0-9_]*\.html)"',
                    r'href="./\1"',
                    content
                )

                html_file.write_text(content, encoding='utf-8')

            except Exception as e:
                print(f"  Warning: Could not clean up {html_file.name}: {e}")

    except Exception as e:
        print(f"Warning: Could not remove version files: {e}")


def flatten_plugin_docs(plugin_output_dir, plugin_folder_name):
    """Flatten the plugin documentation structure by moving files up from nested directories."""
    try:
        nested_path = plugin_output_dir / "videosdk" / "plugins" / plugin_folder_name

        if nested_path.exists():
            for html_file in nested_path.glob("*.html"):
                target_file = plugin_output_dir / html_file.name
                if target_file.exists():
                    target_file.unlink()
                html_file.rename(target_file)

            shutil.rmtree(plugin_output_dir / "videosdk")
        else:
            print(f"Nested path not found for {plugin_folder_name}: {nested_path}")

    except Exception as e:
        print(f"Error flattening docs for {plugin_folder_name}: {e}")


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
        else:
            print(f"Nested agents path not found: {nested_path}")

    except Exception as e:
        print(f"Error flattening agents docs: {e}")


def generate_root_index(docs_dir, base_url=""):
    """Generate a root index.html file that serves as a landing page for all documentation."""
    try:
        available_docs = []

        agents_dir = docs_dir / "agents"
        if agents_dir.exists():
            available_docs.append(
                ("agents", "VideoSDK Agents", "Core agent framework and utilities"))

        for item in docs_dir.iterdir():
            if item.is_dir() and item.name.startswith("plugins-"):
                plugin_name = item.name.replace("plugins-", "")
                display_name = plugin_name.replace("_", " ").title()
                available_docs.append(
                    (item.name, f"VideoSDK {display_name} Plugin", f"Plugin for {display_name} integration"))

        available_docs.sort(key=lambda x: x[0])

        for doc_path, title, description in available_docs:
            print(f"  - {doc_path}: {title}")

        if not available_docs:
            print("Warning: No documentation sections found!")

        if base_url:
            link_format = f"{base_url}/{{}}/"
        else:
            link_format = "./{}/"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VideoSDK Agent SDK Reference</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
            text-align: center;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            text-align: center;
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        .docs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        .doc-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            text-decoration: none;
            color: inherit;
            border: 1px solid #e9ecef;
        }}
        .doc-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }}
        .doc-card h3 {{
            margin: 0 0 0.5rem 0;
            color: #495057;
            font-size: 1.3rem;
        }}
        .doc-card p {{
            margin: 0;
            color: #6c757d;
            font-size: 0.95rem;
        }}
        .doc-card .arrow {{
            float: right;
            color: #007bff;
            font-size: 1.2rem;
            margin-top: -0.5rem;
        }}
        .footer {{
            margin-top: 3rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
        }}
        .footer a {{
            color: #007bff;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>VideoSDK Agent SDK Reference</h1>
            <p>Complete API documentation for VideoSDK agents and plugins</p>
        </div>
    </div>
    
    <div class="container">
        <div class="docs-grid">
"""

        for doc_path, title, description in available_docs:
            html_content += f"""            <a href="{link_format.format(doc_path)}" class="doc-card">
                <h3>{title} <span class="arrow">→</span></h3>
                <p>{description}</p>
            </a>
"""

        html_content += """        </div>
        
        <div class="footer">
            <p>Generated automatically by <a href="https://github.com/videosdk-live/videosdk-agents" target="_blank">VideoSDK Agents</a></p>
        </div>
    </div>
</body>
</html>"""

        index_file = docs_dir / "index.html"
        index_file.write_text(html_content, encoding='utf-8')

    except Exception as e:
        print(f"Error generating root index.html: {e}")


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
        ]

        cmd.append(module_path)

        if name == "rnnoise":
            so_backups = []

            rnnoise_py = path / "rnnoise.py"
            if rnnoise_py.exists():
                backup_name = rnnoise_py.with_suffix('.py.backup')
                rnnoise_py.rename(backup_name)
                so_backups.append((rnnoise_py, backup_name))
                mock_rnnoise_content = '''"""
RNNoise module.
"""

class RNN:
    
    def __init__(self):
        """Initialize RNN instance."""
        pass
    
    def process_frame(self, inbuf):
        """Process frame method."""
        pass
    
    def destroy(self):
        """Destroy method."""
        pass

# Library object
lib = type('MockLib', (), {
    'rnnoise_process_frame': lambda *args: 0.0,
    'rnnoise_create': lambda *args: None,
    'rnnoise_destroy': lambda *args: None
})()
lib.__doc__ = "Library object."
'''
                rnnoise_py.write_text(mock_rnnoise_content)
        else:
            so_backups = []

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=working_dir
        )

        if result.returncode == 0:
            remove_version_files(output_dir)

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
    finally:
        if "plugins" in str(path) and name == "rnnoise" and "so_backups" in locals():
            try:
                for file_path, backup_path in so_backups:
                    if Path(backup_path).exists():
                        Path(backup_path).rename(file_path)
            except Exception as restore_error:
                print(f"    Warning: Could not restore files: {restore_error}")


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
        # Convert hyphens to underscores for the folder path (Python module naming)
        plugin_folder_name = plugin_name.replace("-", "_")
        plugin_path = plugin_dir / "videosdk" / "plugins" / plugin_folder_name

        if plugin_path.exists():
            output_dir = docs_dir / f"plugins-{plugin_name}"
            success = build_docs_for_path(
                plugin_path, output_dir, plugin_folder_name, python_executable)

            if success:
                print(f"Successfully built documentation for {plugin_name}")
            else:
                print(f"Failed to build documentation for {plugin_name}")


def main():
    """Build documentation for all packages."""
    parser = argparse.ArgumentParser(
        description='Build VideoSDK Agent SDK documentation'
    )
    parser.add_argument('--base-url', '-b', default='')
    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    python_executable = get_python_executable()
    ensure_pdoc_installed(python_executable)

    docs_dir = root_dir / "agent-sdk-reference"
    docs_dir.mkdir(parents=True, exist_ok=True)

    build_agents_docs(root_dir, docs_dir, python_executable)
    build_plugin_docs(root_dir, docs_dir, python_executable)

    generate_root_index(docs_dir, args.base_url)

if __name__ == "__main__":
    main()

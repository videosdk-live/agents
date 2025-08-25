import click
import os
import sys
import subprocess
import time
import json
from pathlib import Path
import requests
import yaml
import urllib3
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.traceback import install as install_rich_traceback

# Suppress InsecureRequestWarning for S3 presigned URLs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Initialize rich console
console = Console()

# Get API URL
VIDEOSDK_API_URL = "https://api.videosdk.live"


class VideoSDKError(Exception):
    """Base exception for VideoSDK CLI errors."""

    pass


class DockerError(VideoSDKError):
    """Exception for Docker-related errors."""

    pass


class APIError(VideoSDKError):
    """Exception for API-related errors."""

    pass


class ValidationError(VideoSDKError):
    """Exception for validation errors."""

    pass


class ConfigurationError(VideoSDKError):
    """Exception for configuration-related errors."""

    pass


class FileError(VideoSDKError):
    """Exception for file-related errors."""

    pass


def print_welcome():
    """Print welcome message with instructions."""
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to VideoSDK CLI![/bold cyan]\n\n"
            "[white]Available Commands:[/white]\n"
            "• [green]videosdk run[/green] - Run your worker locally\n"
            "• [green]videosdk deploy[/green] - Deploy your worker to VideoSDK Cloud\n\n"
            "[yellow]Note:[/yellow] Configuration is managed through videosdk.yaml",
            title="VideoSDK CLI",
            border_style="cyan",
        )
    )

def cleanup_container(container_name):
    """Stop and remove a Docker container."""
    try:
        # Check if container exists
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if container_name not in result.stdout:
            return  # Container doesn't exist

        # Stop container if running
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            console.print(
                f"[cyan]✓[/cyan] Stopped container [cyan]{container_name}[/cyan]"
            )
        except subprocess.TimeoutExpired:
            console.print(
                f"[yellow]⚠[/yellow] Container [cyan]{container_name}[/cyan] stop timed out"
            )
        except subprocess.CalledProcessError:
            console.print(
                f"[yellow]⚠[/yellow] Container [cyan]{container_name}[/cyan] was not running"
            )

        # Remove container
        try:
            subprocess.run(
                ["docker", "rm", container_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            console.print(
                f"[cyan]✓[/cyan] Removed container [cyan]{container_name}[/cyan]"
            )
        except subprocess.TimeoutExpired:
            console.print(
                f"[yellow]⚠[/yellow] Container [cyan]{container_name}[/cyan] removal timed out"
            )
        except subprocess.CalledProcessError:
            console.print(
                f"[yellow]⚠[/yellow] Container [cyan]{container_name}[/cyan] could not be removed"
            )
    except Exception as e:
        console.print(
            f"[red]✗[/red] Error cleaning up container [cyan]{container_name}[/cyan]: {str(e)}"
        )


def get_headers(config: dict) -> dict:
    """Get headers for VideoSDK API requests."""
    auth_token = config.get("secrets", {}).get("VIDEOSDK_AUTH_TOKEN")
    if not auth_token:
        raise ValidationError(
            "VIDEOSDK_AUTH_TOKEN is required in videosdk.yaml secrets section"
        )
    return {"Authorization": f"{auth_token}", "Content-Type": "application/json"}


def handle_docker_error(error: subprocess.CalledProcessError) -> None:
    """Handle Docker command errors with user-friendly messages."""
    error_msg = error.stderr.decode() if error.stderr else str(error)

    if "permission denied" in error_msg.lower():
        raise DockerError(
            "Docker permission denied.\n"
            "Please ensure you have the necessary permissions to run Docker commands.\n"
            "You might need to run 'sudo usermod -aG docker $USER' and log out and back in."
        )
    elif "no such file or directory" in error_msg.lower():
        raise DockerError(
            "Docker command not found.\n"
            "Please ensure Docker is installed and in your PATH.\n"
            "Visit https://docs.docker.com/get-docker/ for installation instructions."
        )
    elif "port is already allocated" in error_msg.lower():
        raise DockerError(
            "Port is already in use.\n"
            "Please free up the port or use a different one.\n"
            "You can find the process using the port with 'lsof -i :<port>'"
        )
    elif "image not found" in error_msg.lower():
        raise DockerError(
            "Docker image not found.\n"
            "Please ensure the image exists and try rebuilding with 'videosdk run'"
        )
    elif "no space left on device" in error_msg.lower():
        raise DockerError(
            "No space left on device.\n"
            "Please free up some disk space and try again.\n"
            "You can use 'docker system prune' to clean up unused Docker resources."
        )
    else:
        raise DockerError(f"Docker operation failed: {error_msg}")


def handle_api_error(response: requests.Response) -> None:
    """Handle API errors with user-friendly messages."""
    try:
        error_data = response.json()
        error_message = error_data.get("message", "Unknown error occurred")
        error_details = error_data.get("details", {})
        error_code = error_data.get("code", "UNKNOWN_ERROR")
    except json.JSONDecodeError:
        error_message = response.text
        error_details = {}
        error_code = "INVALID_RESPONSE"

    if response.status_code == 401:
        raise APIError(
            "Authentication failed.\n"
            "Please check your VIDEOSDK_AUTH_TOKEN.\n"
            "You can get your token from the VideoSDK dashboard.\n\n"
            f"Server Error: {error_message}"
        )
    elif response.status_code == 404:
        raise APIError(
            "Resource not found.\n"
            "Please check the deployment ID and try again.\n"
            "You can verify your deployment ID in videosdk.yaml\n\n"
            f"Server Error: {error_message}"
        )
    elif response.status_code == 403:
        raise APIError(
            "Access denied.\n"
            "Please check your permissions and ensure your VIDEOSDK_AUTH_TOKEN is valid.\n\n"
            f"Server Error: {error_message}"
        )
    elif response.status_code >= 500:
        raise APIError(
            "Server error.\n"
            "Please try again later or contact VideoSDK support if the issue persists.\n\n"
            f"Server Error Code: {error_code}\n"
            f"Server Error Message: {error_message}\n"
            f"Error Details: {json.dumps(error_details, indent=2) if error_details else 'None'}"
        )
    elif response.status_code >= 400:
        raise APIError(
            f"API Error:\n"
            f"Status Code: {response.status_code}\n"
            f"Error Code: {error_code}\n"
            f"Error Message: {error_message}\n"
            f"Error Details: {json.dumps(error_details, indent=2) if error_details else 'None'}"
        )
    else:
        raise APIError(f"Unexpected API response: {error_message}")


def validate_environment() -> None:
    """Validate the environment setup."""
    # Check Docker installation
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        raise ValidationError(
            "Docker is not properly installed or not running.\n"
            "Please ensure Docker is installed and the Docker daemon is running.\n"
            "Visit https://docs.docker.com/get-docker/ for installation instructions."
        )
    except FileNotFoundError:
        raise ValidationError(
            "Docker is not installed.\n"
            "Please install Docker first.\n"
            "Visit https://docs.docker.com/get-docker/ for installation instructions."
        )


def validate_env_file(env_path: Path) -> None:
    """Validate the environment file."""
    if not env_path.exists():
        return

    try:
        with open(env_path, "r") as f:
            content = f.read().strip()
            if not content:
                raise ValidationError(
                    f"Environment file {env_path} is empty.\n"
                    "Please add your environment variables or remove the file."
                )

            # Basic validation of .env file format
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" not in line:
                        raise ValidationError(
                            f"Invalid environment variable format in {env_path}:\n"
                            f"Line: {line}\n"
                            "Expected format: KEY=VALUE"
                        )
    except Exception as e:
        raise ValidationError(f"Error reading environment file {env_path}: {str(e)}")


def validate_build_files(main_file: Path, requirement_path: Path) -> None:
    """Validate that all required files exist and are valid."""
    try:
        # Check main.py
        if not main_file.exists():
            raise FileError(
                f"Could not find your main file at {main_file}\n"
                f"Please ensure the path in videosdk.yaml is correct."
            )

        if not main_file.name == "main.py":
            raise FileError(
                "Your main file must be named main.py\n" f"Found: {main_file.name}"
            )

        # Check requirements.txt (optional)
        if requirement_path.exists():
            if not requirement_path.name == "requirements.txt":
                raise FileError(
                    "Your requirements file must be named requirements.txt\n"
                    f"Found: {requirement_path.name}"
                )

            # Validate requirements.txt content if it exists
            try:
                with open(requirement_path, "r") as f:
                    requirements = f.read().strip()
                    if not requirements:
                        raise FileError(
                            "Your requirements.txt file is empty.\n"
                            "Please add your Python dependencies to requirements.txt"
                        )
            except Exception as e:
                raise FileError(f"Could not read requirements.txt: {str(e)}")
    except FileError as e:
        raise e
    except Exception as e:
        raise FileError(f"Unexpected error during file validation: {str(e)}")


def create_dockerfile(directory: Path, entry_point: str = "main.py") -> Path:
    """Create a Dockerfile in the specified directory if it doesn't exist."""
    dockerfile_path = directory / "Dockerfile"

    if not dockerfile_path.exists():
        # Check if requirements.txt exists
        requirements_exist = (directory / "requirements.txt").exists()

        requirements_install = (
            """COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt"""
            if requirements_exist
            else "# No requirements.txt found, skipping dependency installation"
        )

        dockerfile_content = f"""FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
{requirements_install}

# Copy deployment code
COPY . .

# Run the deployment
CMD ["python", "{entry_point}"]
"""
        try:
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            console.print(f"[cyan]✓[/cyan] Created Dockerfile in {directory}")
        except Exception as e:
            raise DockerError(f"Failed to create Dockerfile: {str(e)}")

    return dockerfile_path


def build_docker_image(
    main_file: Path, requirement_path: Path, worker_id: str, save_tar: bool = False
) -> str:
    """Build Docker image for the worker and return the path to the saved image or image name."""
    try:
        # Create Dockerfile in current directory if it doesn't exist
        dockerfile_path = Path.cwd() / "Dockerfile"
        if not dockerfile_path.exists():
            # Get the relative path of the main file from the current directory
            main_file_rel = main_file.relative_to(Path.cwd())

            # Check if requirements.txt exists
            requirements_exist = requirement_path.exists()

            requirements_install = (
                """COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt"""
                if requirements_exist
                else "# No requirements.txt found, skipping dependency installation"
            )

            dockerfile_content = f"""FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
{requirements_install}

# Copy deployment code
COPY . .

# Run the deployment
CMD ["python", "{main_file_rel}"]
"""
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            console.print(f"[cyan]✓[/cyan] Created Dockerfile in {dockerfile_path}")

        # Build Docker image
        image_name = f"videosdk-worker-{worker_id}"
        build_cmd = [
            "docker",
            "build",
            "-t",
            image_name,
            "--platform",
            "linux/arm64",
            "--build-arg",
            "BUILDPLATFORM=linux/arm64",
            ".",
        ]

        try:
            result = subprocess.run(
                build_cmd, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            handle_docker_error(e)

        if save_tar:
            # Create a temporary directory for the tar file
            import tempfile

            temp_dir = tempfile.mkdtemp()
            image_path = Path(temp_dir) / f"{image_name}.tar"

            try:
                save_cmd = ["docker", "save", "-o", str(image_path), image_name]
                subprocess.run(save_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                handle_docker_error(e)
            return str(image_path)
        else:
            return image_name

    except Exception as e:
        if isinstance(e, VideoSDKError):
            raise e
        raise DockerError(f"Failed to build Docker image: {str(e)}")


def validate_worker_id(worker_id: str) -> None:
    """Validate deployment ID format."""
    if not worker_id:
        raise ValidationError("Deployment ID cannot be empty")
   
    # Check for valid characters (alphanumeric, hyphen, underscore)
    if not all(c.isalnum() or c in "-_" for c in worker_id):
        raise ValidationError(
            "Deployment ID can only contain letters, numbers, hyphens, and underscores"
        )

    # Check length (between 3 and 64 characters)
    if len(worker_id) < 3 or len(worker_id) > 64:
        raise ValidationError(
            "Deployment ID must be between 3 and 64 characters long"
        )

def create_yaml_interactive() -> dict:
    """Create videosdk.yaml interactively by asking user for details."""
    console.print(
        Panel.fit(
            "[yellow]No videosdk.yaml found in current directory[/yellow]\n\n"
            "Let's create one! I'll guide you through the process.",
            title="Configuration Setup",
            border_style="yellow",
        )
    )

    # Initialize config with defaults
    config = {
        "version": "1.0",
        "deployment": {
            "id": "",
            "entry": {"path": ""},
            "signaling_base_url": "api.videosdk.live",
        },
        "env": {"path": "./.env"},
        "secrets": {"VIDEOSDK_AUTH_TOKEN": ""},
        "deploy": {"cloud": True},
    }

    # Ask for worker details
    console.print("\n[bold cyan]Step 1: Deployment Configuration[/bold cyan]")
    console.print("[dim]This section defines your deployment's basic settings.[/dim]")
    console.print("[dim]Deployment ID must be 3-64 characters long and can only contain letters, numbers, hyphens, and underscores.[/dim]")
    
    config["deployment"]["id"] = click.prompt("Deployment ID")
    validate_worker_id(config["deployment"]["id"])
    
    # Ask for entry point
    default_entry = "src/main.py"
    console.print("[dim]Specify the path to your main Python file.[/dim]")
    entry_path = click.prompt(
        "Path to main Python file",
        default=default_entry
    )
    config["deployment"]["entry"]["path"] = entry_path
    
    # Ask for environment file path
    console.print("\n[bold cyan]Step 2: Environment Configuration[/bold cyan]")
    console.print("[dim]Configure your deployment's environment file.[/dim]")
    console.print("[dim]Specify the path to your .env file (optional).[/dim]")

    env_path = click.prompt("Path to environment file", default="./.env")
    config["env"]["path"] = env_path

    # Validate environment file if it exists
    if Path(env_path).exists():
        validate_env_file(Path(env_path))

    # Ask for secrets
    console.print("\n[bold cyan]Step 3: Secrets Configuration[/bold cyan]")
    console.print("[dim]Configure your deployment's secrets.[/dim]")
    console.print("[dim]Enter your VideoSDK authentication token.[/dim]")

    secrets = config["secrets"]
    secrets["VIDEOSDK_AUTH_TOKEN"] = click.prompt("VideoSDK Auth Token")

    # Ask for deployment settings
    console.print("\n[bold cyan]Step 4: Deployment Configuration[/bold cyan]")
    console.print("[dim]Configure your deployment settings.[/dim]")
    console.print("[dim]Choose whether to enable cloud deployment.[/dim]")

    deploy = config["deploy"]
    deploy["cloud"] = click.confirm("Enable cloud deployment?", default=True)

    # Save the configuration
    config_path = Path.cwd() / "videosdk.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(
            Panel.fit(
                f"[green]Success![/green] Created videosdk.yaml at:\n"
                f"[cyan]{config_path}[/cyan]\n\n"
                f"You can now run your worker with:\n"
                f"[green]videosdk run[/green]",
                title="Configuration Complete",
                border_style="green",
            )
        )
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to create videosdk.yaml: {str(e)}")


def load_config() -> dict:
    """Load configuration from videosdk.yaml file."""
    config_path = Path.cwd() / "videosdk.yaml"
    if not config_path.exists():
        # Create config interactively
        return create_yaml_interactive()

    try:
        with open(config_path, "r") as f:
            content = f.read().strip()
            if not content:
                raise ConfigurationError(
                    "videosdk.yaml is empty.\n"
                    "Please add configuration or run 'videosdk run' to create it interactively."
                )

            try:
                config = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ConfigurationError(
                    f"Invalid YAML in videosdk.yaml:\n{str(e)}\n"
                    "Please check the file format and try again."
                )

        # Validate that config is a dictionary
        if not isinstance(config, dict):
            raise ConfigurationError(
                f"Invalid configuration format in videosdk.yaml.\n"
                f"Expected a dictionary, got {type(config).__name__}.\n"
                "Please check the file format and try again."
            )

        # Validate required fields
        if not config:
            raise ConfigurationError(
                "videosdk.yaml is empty.\n"
                "Please add configuration or run 'videosdk run' to create it interactively."
            )

        # Validate version
        version = config.get("version")
        if version != "1.0":
            raise ConfigurationError(
                f"Unsupported configuration version.\n"
                f"Expected version 1.0, found: {version}\n"
                "Please update your configuration to use version 1.0."
            )

        # Validate worker section
        deployment = config.get('deployment')
        if not deployment:
            raise ConfigurationError(
                "Missing 'deployment' section in videosdk.yaml.\n"
                "Please add a deployment section with required fields:\n"
                "deployment:\n"
                "  id: your-deployment-id\n"
                "  entry:\n"
                "    path: path/to/main.py"
            )
        
        if not isinstance(deployment, dict):
            raise ConfigurationError(
                f"Invalid deployment section format.\n"
                f"Expected a dictionary, got {type(deployment).__name__}.\n"
                "Please check the deployment section format in videosdk.yaml."
            )
        
        worker_id = deployment.get('id')
        if not worker_id:
            raise ConfigurationError(
                "Missing 'deployment.id' in videosdk.yaml.\n"
                "Please add a unique identifier for your deployment:\n"
                "deployment:\n"
                "  id: your-deployment-id"
            )

        try:
            validate_worker_id(worker_id)
        except ValidationError as e:
            raise ConfigurationError(f"Invalid deployment ID in videosdk.yaml: {str(e)}")
        
        entry = deployment.get('entry', {})
        if not isinstance(entry, dict):
            raise ConfigurationError(
                f"Invalid entry section format.\n"
                f"Expected a dictionary, got {type(entry).__name__}.\n"
                "Please check the entry section format in videosdk.yaml."
            )

        entry_path = entry.get("path")
        if not entry_path:
            raise ConfigurationError(
                "Missing 'deployment.entry.path' in videosdk.yaml.\n"
                "Please specify the path to your main Python file:\n"
                "deployment:\n"
                "  entry:\n"
                "    path: path/to/main.py"
            )

        # Validate signaling base URL
        signaling_base_url = worker.get("signaling_base_url", "api.videosdk.live")
        if not signaling_base_url:
            raise ConfigurationError(
                "Missing 'worker.signaling_base_url' in videosdk.yaml.\n"
                "Please specify the VideoSDK signaling base URL:\n"
                "worker:\n"
                "  signaling_base_url: api.videosdk.live"
            )

        # Validate env section
        env = config.get("env", {})
        if not isinstance(env, dict):
            raise ConfigurationError(
                f"Invalid env section format.\n"
                f"Expected a dictionary, got {type(env).__name__}.\n"
                "Please check the env section format in videosdk.yaml."
            )
        env.setdefault("path", "./.env")

        # Validate environment file if it exists
        env_path = Path(env["path"])
        if env_path.exists():
            validate_env_file(env_path)

        # Validate secrets section
        secrets = config.get("secrets", {})
        if not isinstance(secrets, dict):
            raise ConfigurationError(
                f"Invalid secrets section format.\n"
                f"Expected a dictionary, got {type(secrets).__name__}.\n"
                "Please check the secrets section format in videosdk.yaml."
            )
        if not secrets.get("VIDEOSDK_AUTH_TOKEN"):
            raise ConfigurationError(
                "Missing 'VIDEOSDK_AUTH_TOKEN' in secrets section.\n"
                "Please add your VideoSDK authentication token:\n"
                "secrets:\n"
                "  VIDEOSDK_AUTH_TOKEN: your-token"
            )

        # Validate deploy section
        deploy = config.get("deploy", {})
        if not isinstance(deploy, dict):
            raise ConfigurationError(
                f"Invalid deploy section format.\n"
                f"Expected a dictionary, got {type(deploy).__name__}.\n"
                "Please check the deploy section format in videosdk.yaml."
            )
        deploy.setdefault("cloud", True)

        return config
    except ConfigurationError as e:
        raise e
    except Exception as e:
        raise ConfigurationError(
            f"Error reading videosdk.yaml:\n{str(e)}\n"
            "Please check the file format and try again.\n"
            "You can run 'videosdk run' to create a new configuration interactively."
        )


def format_log_line(line: str, color: str = None) -> str:
    """Format a log line with appropriate colors and styling."""
    # Remove ANSI color codes from the line
    import re

    line = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)

    # Try to detect log level and format accordingly
    line_lower = line.lower()
    if any(
        level in line_lower for level in ["error", "exception", "failed", "failure"]
    ):
        return f"[red]{line}[/red]"
    elif any(level in line_lower for level in ["warning", "warn"]):
        return f"[yellow]{line}[/yellow]"
    elif any(level in line_lower for level in ["info", "information"]):
        return f"[cyan]{line}[/cyan]"
    elif any(level in line_lower for level in ["debug"]):
        return f"[dim]{line}[/dim]"
    elif color:
        return f"[{color}]{line}[/{color}]"
    else:
        return line


def read_output(pipe, color=None):
    """Read and format output from a pipe with appropriate colors."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                line = line.strip()
                if line:  # Only print non-empty lines
                    formatted_line = format_log_line(line, color)
                    console.print(formatted_line)
    except Exception as e:
        console.print(f"[red]Error reading output: {str(e)}[/red]")


@click.group()
def cli():
    """VideoSDK Agents CLI - Run and deploy your workers using videosdk.yaml configuration"""
    print_welcome()


@cli.command()
def run():
    """Run your worker locally in a Docker container using videosdk.yaml configuration"""
    try:
        validate_environment()

        # Load configuration
        config = load_config()
        worker = config['deployment']
        
        # Use absolute paths for all file operations
        main_file = Path(worker["entry"]["path"]).resolve()
        requirement_path = Path("requirements.txt").resolve()
        current_dir = Path.cwd().resolve()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Starting worker [cyan]{worker['id']}[/cyan]...", total=None
            )

            # Ensure we're in the correct directory
            if main_file.parent != current_dir:
                console.print(
                    f"[cyan]ℹ[/cyan] Using main file from [cyan]{main_file.parent}[/cyan], building from [cyan]{current_dir}[/cyan]"
                )

            validate_build_files(main_file, requirement_path)

            # Build Docker image
            try:
                progress.update(
                    task, description=f"Building worker [cyan]{worker['id']}[/cyan]..."
                )
                image_name = build_docker_image(
                    main_file, requirement_path, worker["id"], save_tar=False
                )

                progress.update(task, description=f"Running worker [cyan]{worker['id']}[/cyan]...")
                container_name = f"videosdk-deployment-{worker['id']}-{int(time.time())}"
                
                # Clear the progress display
                progress.stop()
                console.clear()
                console.print(Panel.fit(
                    f"[bold cyan]Deployment [cyan]{worker['id']}[/cyan] is running[/bold cyan]\n"
                    f"Container: [cyan]{container_name}[/cyan]\n"
                    f"Press [yellow]Ctrl+C[/yellow] to stop",
                    border_style="cyan"
                ))
                console.print("[dim]Deployment logs:[/dim]\n")
               
                # Run the container and stream logs directly
                run_cmd = [
                    "docker",
                    "run",
                    "--name",
                    container_name,
                    "--rm",
                ]

                # Add environment variables from .env file if it exists
                env_path = Path(config["env"]["path"]).resolve()
                if env_path.exists():
                    console.print(
                        f"[cyan]ℹ[/cyan] Using environment variables from [cyan]{env_path}[/cyan]"
                    )
                    run_cmd.extend(["--env-file", str(env_path)])

                # Add secrets from config
                for key, value in config.get("secrets", {}).items():
                    run_cmd.extend(["-e", f"{key}={value}"])

                # Add signaling base URL from config
                signaling_base_url = worker.get(
                    "signaling_base_url", "api.videosdk.live"
                )
                run_cmd.extend(
                    ["-e", f"VIDEOSDK_SIGNALING_BASE_URL={signaling_base_url}"]
                )

                # Add the image name at the end
                run_cmd.append(image_name)

                try:
                    # Run the container and stream output directly
                    process = subprocess.Popen(
                        run_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    # Start threads to read stdout and stderr
                    import threading

                    stdout_thread = threading.Thread(
                        target=read_output, args=(process.stdout,)
                    )
                    stderr_thread = threading.Thread(
                        target=read_output, args=(process.stderr, "red")
                    )

                    stdout_thread.daemon = True
                    stderr_thread.daemon = True

                    # Start threads and wait for process
                    stdout_thread.start()
                    stderr_thread.start()

                    # Wait for the process to complete
                    return_code = process.wait()

                    # Wait for output threads to complete with timeout
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)

                    if return_code != 0:
                        # If the container failed, show its logs
                        console.print(
                            "\n[yellow]⚠[/yellow] Container failed. Showing logs:"
                        )
                        log_cmd = ["docker", "logs", container_name]
                        try:
                            log_process = subprocess.run(
                                log_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False,
                                timeout=10,
                            )
                            if log_process.stdout:
                                for line in log_process.stdout.splitlines():
                                    if line.strip():
                                        console.print(format_log_line(line))
                            if log_process.stderr:
                                for line in log_process.stderr.splitlines():
                                    if line.strip():
                                        console.print(format_log_line(line, "red"))
                        except subprocess.TimeoutExpired:
                            console.print(
                                "[red]✗[/red] Could not retrieve container logs: timeout"
                            )
                        except Exception as e:
                            console.print(
                                f"[red]✗[/red] Could not retrieve container logs: {str(e)}"
                            )

                        raise DockerError(
                            f"Container failed with exit code {return_code}"
                        )

                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully
                    console.print("\n[yellow]⚠[/yellow] Stopping deployment...")
                    cleanup_container(container_name)
                    raise click.Abort()
                except Exception as e:
                    cleanup_container(container_name)
                    raise DockerError(f"Error running container: {str(e)}")

            except VideoSDKError as e:
                raise e
            except Exception as e:
                raise DockerError(f"Error running deployment: {str(e)}")
    except VideoSDKError as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {str(e)}")
        sys.exit(1)


@cli.command()
def deploy():
    """Deploy your worker to VideoSDK using videosdk.yaml configuration"""
    try:
        validate_environment()

        # Load configuration
        config = load_config()
        worker = config['deployment']
        main_file = Path(worker['entry']['path']).resolve()
        requirement_path = Path('requirements.txt').resolve()
        
        # Check if deployment is allowed
        deploy_config = config.get("deploy", {})
        if not deploy_config.get(
            "cloud", True
        ):  # Changed default to match other functions
            raise ConfigurationError(
                "Deployment is disabled in videosdk.yaml.\n"
                "To enable deployment, set:\n"
                "deploy:\n"
                "  cloud: true"
            )
        
        if not worker.get('id'):
            raise ValidationError("deployment.id is required in videosdk.yaml for deployment")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Step 0: Validate files
            task = progress.add_task(f"Checking deployment {worker['id']}...", total=100)
            validate_build_files(main_file, requirement_path)
            progress.update(task, completed=100)

            # Step 1: Get deployment URL
            task = progress.add_task(
                f"Getting deployment details for {worker['id']}...", total=100
            )
            deployment_url = (
                f"{VIDEOSDK_API_URL}/ai/v1/ai-workers/{worker['id']}/deployments"
            )

            try:
                response = requests.post(deployment_url, headers=get_headers(config))
                if response.status_code >= 400:
                    handle_api_error(response)
                deployment_data = response.json()
                presigned_url = deployment_data.get("presignedUrl")
                if not presigned_url:
                    raise APIError(
                        "Could not get deployment details.\n"
                        "Server response did not contain a presigned URL.\n"
                        f"Response: {json.dumps(deployment_data, indent=2)}"
                    )
                progress.update(task, completed=100)
            except requests.exceptions.RequestException as e:
                raise APIError(
                    f"Could not connect to VideoSDK API.\n"
                    f"URL: {deployment_url}\n"
                    f"Error: {str(e)}"
                )

            # Step 2: Build Docker image
            task = progress.add_task(f"Building deployment {worker['id']}...", total=100)
            try:
                docker_image_path = build_docker_image(
                    main_file, requirement_path, worker["id"], save_tar=True
                )
                progress.update(task, completed=100)
            except VideoSDKError as e:
                raise e
            except Exception as e:
                raise DockerError(f"Could not prepare your deployment: {str(e)}")

            # Step 3: Upload to S3
            task = progress.add_task(f"Uploading deployment {worker['id']}...", total=100)
            try:
                with open(docker_image_path, "rb") as f:
                    file_size = os.path.getsize(docker_image_path)
                    console.print(
                        f"[dim]Uploading {file_size / (1024*1024):.2f} MB...[/dim]"
                    )

                    # Create a wrapper for the file to track progress
                    class ProgressFile:
                        def __init__(self, file, progress, task):
                            self.file = file
                            self.progress = progress
                            self.task = task
                            self.bytes_read = 0
                            self.total_size = file_size

                        def read(self, size=-1):
                            data = self.file.read(size)
                            if data:
                                self.bytes_read += len(data)
                                # Update progress as percentage
                                percentage = (self.bytes_read / self.total_size) * 100
                                self.progress.update(self.task, completed=percentage)
                            return data

                        def __enter__(self):
                            return self

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            self.file.close()

                    # Use the progress file wrapper for upload
                    with ProgressFile(f, progress, task) as progress_file:
                        try:
                            session = requests.Session()
                            session.verify = False
                            upload_response = session.put(
                                presigned_url,
                                data=progress_file,
                                timeout=600,
                                headers={
                                    "Content-Type": "application/x-tar",
                                    "Content-Length": str(file_size),
                                },
                            )
                        except requests.exceptions.SSLError as e:
                            raise APIError(
                                "SSL verification failed during upload.\n"
                                "This could be due to:\n"
                                "1. Invalid SSL certificate\n"
                                "2. Network proxy issues\n"
                                "3. System time being incorrect\n"
                                "4. Python SSL library issues\n\n"
                                f"Error details: {str(e)}\n\n"
                                "Try running:\n"
                                "pip install --upgrade certifi requests urllib3"
                            )
                        except requests.exceptions.ConnectionError as e:
                            raise APIError(
                                "Connection error during upload.\n"
                                "Please check your internet connection and try again.\n\n"
                                f"Error details: {str(e)}"
                            )
                        except requests.exceptions.Timeout as e:
                            raise APIError(
                                "Upload timed out.\n"
                                "The file might be too large or the connection too slow.\n"
                                "Please try again.\n\n"
                                f"Error details: {str(e)}"
                            )
                        except Exception as e:
                            raise APIError(
                                f"Unexpected error during upload: {str(e)}\n"
                                "Please try again or contact support if the issue persists."
                            )

                    if upload_response.status_code not in [200, 204]:
                        try:
                            error_data = upload_response.json()
                            error_message = error_data.get(
                                "message", "Unknown error occurred"
                            )
                            error_details = error_data.get("details", {})
                            raise APIError(
                                "Could not upload your deployment.\n"
                                f"Status Code: {upload_response.status_code}\n"
                                f"Error Message: {error_message}\n"
                                f"Error Details: {json.dumps(error_details, indent=2) if error_details else 'None'}"
                            )
                        except json.JSONDecodeError:
                            raise APIError(
                                "Could not upload your deployment.\n"
                                f"Status Code: {upload_response.status_code}\n"
                                f"Response: {upload_response.text}"
                            )
                progress.update(task, completed=100)
            except Exception as e:
                raise APIError(f"Could not upload your deployment: {str(e)}")
            finally:
                # Cleanup: Remove the temporary tar file and its directory
                try:
                    import shutil

                    shutil.rmtree(Path(docker_image_path).parent)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not clean up temporary files: {str(e)}[/yellow]"
                    )

        # Print success message
        console.print(Panel.fit(
            f"[bold green]Success![/bold green]\n\n"
            f"Your deployment has been deployed successfully!\n"
            f"Deployment ID: [cyan]{worker['id']}[/cyan]\n\n"
            f"[yellow]Note:[/yellow] It may take a few minutes for your deployment to be fully available on the cloud.\n"
            f"Please wait while we process your deployment.",
            title="Deployment Complete",
            border_style="green"
        ))
    except VideoSDKError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except VideoSDKError as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {str(e)}")
        console.print(
            "\n[yellow]If this error persists, please contact VideoSDK support.[/yellow]"
        )
        sys.exit(1)

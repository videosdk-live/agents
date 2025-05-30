import click
import os
import sys
import signal
import subprocess
import socket
import time
import json
import asyncio
import inspect
from pathlib import Path
from dotenv import load_dotenv
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Get environment variables
VIDEOSDK_AUTH_TOKEN = os.getenv("VIDEOSDK_AUTH_TOKEN")
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

def print_welcome():
    """Print welcome message with instructions."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to VideoSDK CLI![/bold cyan]\n\n"
        "[white]Quick Start:[/white]\n"
        "1. Run your agent: [green]videosdk run[/green]\n"
        "2. Deploy your agent: [green]videosdk deploy --agent-id your-id[/green]\n\n"
        "[yellow]Note:[/yellow] Make sure you have set your VIDEOSDK_AUTH_TOKEN environment variable.",
        title="VideoSDK CLI",
        border_style="cyan"
    ))

def ensure_valid_agent_path(agent_path: str):
    """Ensure the main file is valid and points to a main.py file."""
    if not agent_path.endswith("main.py"):
        raise click.ClickException("Your main file must be named main.py")
    return agent_path

def cleanup_container(container_name):
    """Stop and remove a Docker container."""
    try:
        subprocess.run(
            ["docker", "stop", container_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        click.secho(f"  - Stopped container '{container_name}'.", fg="white")
        try:
            subprocess.run(
                ["docker", "rm", container_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            click.secho(f"  - Removed container '{container_name}'.", fg="white")
        except subprocess.CalledProcessError:
            click.secho(
                f"Container '{container_name}' could not be removed or does not exist.",
                fg="red",
                bold=True,
            )
    except subprocess.CalledProcessError:
        click.secho(f"Container '{container_name}' was not running.", fg="yellow")

def print_container_logs(container_name: str):
    """Retrieve and print the container logs for debugging."""
    click.secho(f"Retrieving logs for container '{container_name}'...", fg="yellow")
    try:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            click.secho("--- Container Logs (STDOUT) ---", fg="red", bold=True)
            click.secho(result.stdout, fg="red")
        if result.stderr:
            click.secho("--- Container Logs (STDERR) ---", fg="red", bold=True)
            click.secho(result.stderr, fg="red")
    except Exception as ex:
        click.secho(
            f"Failed to retrieve logs for container '{container_name}': {ex}",
            fg="red",
            bold=True,
        )

def get_headers():
    """Get headers for VideoSDK API requests."""
    if not VIDEOSDK_AUTH_TOKEN:
        raise click.ClickException("VIDEOSDK_AUTH_TOKEN environment variable is required")
    return {
        "Authorization": f"{VIDEOSDK_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

def handle_docker_error(error: subprocess.CalledProcessError) -> None:
    """Handle Docker command errors with user-friendly messages."""
    error_msg = error.stderr.decode() if error.stderr else str(error)
    
    if "permission denied" in error_msg.lower():
        raise DockerError("Docker permission denied. Please ensure you have the necessary permissions to run Docker commands.")
    elif "no such file or directory" in error_msg.lower():
        raise DockerError("Docker command not found. Please ensure Docker is installed and in your PATH.")
    elif "port is already allocated" in error_msg.lower():
        raise DockerError("Port is already in use. Please free up the port or use a different one.")
    elif "image not found" in error_msg.lower():
        raise DockerError("Docker image not found. Please ensure the image exists.")
    else:
        raise DockerError(f"Docker operation failed: {error_msg}")

def handle_api_error(response: requests.Response) -> None:
    """Handle API errors with user-friendly messages."""
    print("Response", response.text)
    if response.status_code == 401:
        raise APIError("Authentication failed. Please check your VIDEOSDK_AUTH_TOKEN")
    elif response.status_code == 404:
        raise APIError("Resource not found. Please check the agent ID and try again.")
    elif response.status_code == 403:
        raise APIError("Access denied. Please check your permissions.")
    elif response.status_code >= 500:
        raise APIError("Server error. Please try again later.")
    elif response.status_code >= 400:
        try:
            error_data = response.json()
            error_message = error_data.get("message", "Unknown error occurred")
            raise APIError(f"API Error: {error_message}")
        except json.JSONDecodeError:
            raise APIError(f"API Error: {response.text}")

def validate_environment(require_auth_token: bool = True) -> None:
    """Validate the environment setup.
    
    Args:
        require_auth_token: Whether to require VIDEOSDK_AUTH_TOKEN (default: True)
    """
    if require_auth_token and not VIDEOSDK_AUTH_TOKEN:
        raise ValidationError("VIDEOSDK_AUTH_TOKEN environment variable is required")
    
    # Check Docker installation
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        raise ValidationError("Docker is not properly installed or not running")
    except FileNotFoundError:
        raise ValidationError("Docker is not installed. Please install Docker first")

def validate_build_files(main_file: Path, requirement_path: Path) -> None:
    """Validate that all required files exist and are valid."""
    try:
        # Check main.py
        if not main_file.exists():
            raise ValidationError(f"Could not find your main file at {main_file}")
        
        if not main_file.name == "main.py":
            raise ValidationError("Your main file must be named main.py")
        
        # Check requirements.txt
        if not requirement_path.exists():
            raise ValidationError(f"Could not find requirements.txt at {requirement_path}")
        
        if not requirement_path.name == "requirements.txt":
            raise ValidationError("Your requirements file must be named requirements.txt")
        
        # Validate requirements.txt content
        try:
            with open(requirement_path, 'r') as f:
                requirements = f.read().strip()
                if not requirements:
                    raise ValidationError("Your requirements.txt file is empty")
        except Exception as e:
            raise ValidationError(f"Could not read requirements.txt: {str(e)}")
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error during validation: {str(e)}")

def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded from Docker build.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file should be excluded, False otherwise
    """
    # List of files and patterns to exclude
    exclude_patterns = [
        # Environment and config files
        '.env',
        '.env.*',
        'config.json',
        'config.yaml',
        'config.yml',
        'settings.json',
        'settings.yaml',
        'settings.yml',
        
        # Version control
        '.git',
        '.gitignore',
        '.gitattributes',
        
        # IDE and editor files
        '.vscode',
        '.idea',
        '*.swp',
        '*.swo',
        '*~',
        
        # Python specific
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        '*.so',
        '*.egg',
        '*.egg-info',
        'dist',
        'build',
        '*.egg-info',
        
        # Logs and databases
        '*.log',
        '*.sqlite',
        '*.db',
        
        # Docker related
        'Dockerfile',
        'docker-compose*.yml',
        'docker-compose*.yaml',
        
        # Documentation
        'README.md',
        'LICENSE',
        'CHANGELOG.md',
        'docs',
        
        # Test files
        'tests',
        'test_*.py',
        '*_test.py',
        
        # Temporary files
        '*.tmp',
        '*.temp',
        '*.bak',
        '*.swp',
        '*.swo',
        '*~'
    ]
    
    # Check if file matches any exclude pattern
    for pattern in exclude_patterns:
        if file_path.match(pattern):
            return True
    
    # Check if file is hidden (starts with .)
    if file_path.name.startswith('.'):
        return True
        
    return False

def create_dockerfile(directory: Path, agent_id: str, entry_point: str = "main.py") -> Path:
    """Create a Dockerfile in the specified directory if it doesn't exist.
    
    Args:
        directory: Directory where to create the Dockerfile
        agent_id: Agent ID for environment variables
        entry_point: Path to the main Python file to run
        
    Returns:
        Path: Path to the created or existing Dockerfile
    """
    dockerfile_path = directory / "Dockerfile"
    
    if not dockerfile_path.exists():
        dockerfile_content = f"""FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Run the agent
CMD ["python", "{entry_point}"]
"""
        try:
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            console.print(f"[green]Created Dockerfile in {directory}[/green]")
        except Exception as e:
            raise DockerError(f"Failed to create Dockerfile: {str(e)}")
    
    return dockerfile_path

def build_docker_image(main_file: Path, requirement_path: Path, agent_id: str, save_tar: bool = False) -> str:
    """Build Docker image for the agent and return the path to the saved image or image name.
    
    Args:
        main_file: Path to the main.py file
        requirement_path: Path to requirements.txt
        agent_id: Agent ID for the image name
        save_tar: Whether to save the image as a tar file (needed for deployment)
    """
    try:
        # Create Dockerfile in current directory if it doesn't exist
        dockerfile_path = Path.cwd() / "Dockerfile"
        if not dockerfile_path.exists():
            # Get the relative path of the main file from the current directory
            main_file_rel = main_file.relative_to(Path.cwd())
            
            dockerfile_content = f"""FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Run the agent
CMD ["python", "{main_file_rel}"]
"""
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            console.print(f"[green]Created Dockerfile in {dockerfile_path}[/green]")

        # Build Docker image
        image_name = f"videosdk-agent-{agent_id}"
        build_cmd = [
            "docker", "build",
            "-t", image_name,
            "--platform", "linux/arm64",
            "--build-arg", "BUILDPLATFORM=linux/arm64",
            "."
        ]

        try:
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True
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

@click.group()
def cli():
    """VideoSDK Agents CLI - Run and deploy your agents"""
    print_welcome()

@cli.command()
@click.option('--entry', default='main.py', help='Path to your main file (default: main.py in current directory)')
@click.option('--requirement', '-r', default='requirements.txt', help='Path to your requirements.txt (default: requirements.txt in current directory)')
def run(entry, requirement):
    """Run your agent locally in a Docker container"""
    try:
        # Don't require auth token for run command
        validate_environment(require_auth_token=False)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Starting your agent...", total=None)
            
            main_file = Path(entry).resolve()
            requirement_path = Path(requirement).resolve()
            
            # Ensure we're in the correct directory
            if main_file.parent != Path.cwd():
                console.print(f"[yellow]Warning:[/yellow] Using main file from {main_file.parent}, but building from current directory")
            
            validate_build_files(main_file, requirement_path)

            # Build Docker image
            try:
                progress.update(task, description="Building your agent...")
                image_name = build_docker_image(main_file, requirement_path, "local", save_tar=False)
                
                # Run the container
                progress.update(task, description="Running your agent...")
                container_name = f"videosdk-agent-{int(time.time())}"
                
                # Clear the progress display
                progress.stop()
                console.clear()
                console.print("[bold cyan]Your agent is running. Press Ctrl+C to stop.[/bold cyan]")
                console.print("[dim]Logs from your agent:[/dim]\n")
                
                # Run the container and stream logs directly
                run_cmd = [
                    "docker", "run",
                    "--name", container_name,
                    "--rm",
                ]
                
                # Add .env file if it exists in current directory
                env_file_path = Path.cwd() / ".env"
                if env_file_path.exists():
                    console.print(f"[green]Using environment variables from {env_file_path}[/green]")
                    run_cmd.extend(["--env-file", str(env_file_path)])
                else:
                    console.print("[yellow]No .env file found. Using default environment variables.[/yellow]")
                
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
                        universal_newlines=True
                    )
                    
                    # Function to read and print output
                    def read_output(pipe, color=None):
                        for line in iter(pipe.readline, ''):
                            if line:
                                line = line.strip()
                                if line:  # Only print non-empty lines
                                    if color:
                                        console.print(f"[{color}]{line}[/{color}]")
                                    else:
                                        console.print(line)
                    
                    # Start threads to read stdout and stderr
                    import threading
                    stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
                    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, 'red'))
                    
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    # Wait for the process to complete
                    return_code = process.wait()
                    
                    # Wait for output threads to complete
                    stdout_thread.join()
                    stderr_thread.join()
                    
                    if return_code != 0:
                        # If the container failed, show its logs
                        console.print("\n[yellow]Container failed. Showing logs:[/yellow]")
                        log_cmd = ["docker", "logs", container_name]
                        try:
                            log_process = subprocess.run(
                                log_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False
                            )
                            if log_process.stdout:
                                console.print(log_process.stdout)
                            if log_process.stderr:
                                console.print(f"[red]{log_process.stderr}[/red]")
                        except Exception as e:
                            console.print(f"[red]Could not retrieve container logs: {str(e)}[/red]")
                        
                        raise DockerError(f"Container failed with exit code {return_code}")
                    
                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully
                    console.print("\n[yellow]Stopping your agent...[/yellow]")
                    cleanup_container(container_name)
                    raise click.Abort()
                except Exception as e:
                    cleanup_container(container_name)
                    raise DockerError(f"Error running container: {str(e)}")
                
            except VideoSDKError as e:
                raise e
            except Exception as e:
                raise DockerError(f"Error running agent: {str(e)}")
    except VideoSDKError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--entry', default='main.py', help='Path to your main file (default: main.py in current directory)')
@click.option('--requirement', '-r', default='requirements.txt', help='Path to your requirements.txt (default: requirements.txt in current directory)')
@click.option('--agent-id', required=True, help='Your agent ID')
def deploy(entry, requirement, agent_id):
    """Deploy your agent to VideoSDK"""
    try:
        validate_environment()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Step 0: Validate files
            task = progress.add_task("Checking your files...", total=100)
            main_file = Path(entry).resolve()
            requirement_path = Path(requirement).resolve()
            validate_build_files(main_file, requirement_path)
            progress.update(task, completed=100)

            # Step 1: Get deployment URL
            task = progress.add_task("Getting deployment details...", total=100)
            deployment_url = f"{VIDEOSDK_API_URL}/ai/v1/ai-agents/{agent_id}/deployments"
            
            try:
                response = requests.post(deployment_url, headers=get_headers())
                handle_api_error(response)
                deployment_data = response.json()
                presigned_url = deployment_data.get("presignedUrl")
                if not presigned_url:
                    raise APIError("Could not get deployment details")
                progress.update(task, completed=100)
            except requests.exceptions.RequestException as e:
                raise APIError(f"Could not connect to VideoSDK: {str(e)}")

            # Step 2: Build Docker image
            task = progress.add_task("Building your agent...", total=100)
            try:
                docker_image_path = build_docker_image(main_file, requirement_path, agent_id, save_tar=True)
                progress.update(task, completed=100)
            except VideoSDKError as e:
                raise e
            except Exception as e:
                raise DockerError(f"Could not prepare your agent: {str(e)}")

            # Step 3: Upload to S3
            task = progress.add_task("Uploading your agent...", total=100)
            try:
                with open(docker_image_path, 'rb') as f:
                    upload_response = requests.put(presigned_url, data=f)
                    if upload_response.status_code not in [200, 204]:
                        raise APIError("Could not upload your agent")
                progress.update(task, completed=100)
            except Exception as e:
                raise APIError(f"Could not upload your agent: {str(e)}")
            finally:
                # Cleanup: Remove the temporary tar file and its directory
                try:
                    import shutil
                    shutil.rmtree(Path(docker_image_path).parent)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not clean up temporary files: {str(e)}[/yellow]")

        # Print success message
        console.print(Panel.fit(
            f"[bold green]Success![/bold green]\n\n"
            f"Your agent has been deployed successfully!\n"
            f"Agent ID: [cyan]{agent_id}[/cyan]",
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
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        sys.exit(1) 
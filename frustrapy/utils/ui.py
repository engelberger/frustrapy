import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from ..analysis.exceptions import FileOperationError, SubprocessError, ValidationError, MissingBackboneAtomError

console = Console()

def display_error(exception: Exception, is_debug: bool = False) -> None:
    """Display a user-friendly error message using Rich."""
    message = Text()
    title = "Error"
    style = "bold red"
    
    if isinstance(exception, FileOperationError):
        title = "File Operation Error"
        message.append(f"Error: {exception.message}\n", style=style)
        if exception.src:
            message.append(f"  Source:      {exception.src}\n")
        if exception.dst:
            message.append(f"  Destination: {exception.dst}\n")
        if not exception.message.startswith("Destination file already exists"):
             # Suggest cautious mode only if it's not an overwrite issue
            message.append("\nSuggestion: Check file paths and permissions.", style="yellow")
    elif isinstance(exception, SubprocessError):
        title = "Subprocess Error"
        message.append(f"Error: {exception.message}\n", style=style)
        if exception.cmd:
            message.append(f"  Command: {' '.join(exception.cmd)}\n")
        if exception.returncode is not None:
            message.append(f"  Return Code: {exception.returncode}\n")
        if is_debug and exception.stderr:
             message.append(f"\n--- Error Output (Debug) ---\n{exception.stderr}", style="dim")
        else:
             message.append("\nSuggestion: Run with debug=True for more details on subprocess output.", style="yellow")
    elif isinstance(exception, ValidationError):
        title = "Configuration Error"
        message.append(f"Error: {exception.message}\n", style=style)
        if exception.param_name:
            message.append(f"  Parameter: {exception.param_name}\n")
        if exception.value is not None:
             message.append(f"  Value:     {exception.value!r}\n")
        message.append("\nSuggestion: Please check your input parameters.", style="yellow")
    elif isinstance(exception, MissingBackboneAtomError):
        title = "Missing Backbone Atom Error"
        message.append(f"Error: {exception.message}\n", style=style)
        if exception.missing_atoms:
            message.append("Missing backbone atoms:\n")
            for atom in exception.missing_atoms:
                if isinstance(atom, tuple) and len(atom) == 3:
                    residue, chain, atom_name = atom
                    message.append(f"  - Residue {residue} (Chain {chain}): missing {atom_name} atom\n")
                else:
                    message.append(f"  - {atom}\n")
        message.append("\nSuggestion: Please check your PDB file for completeness or repair missing atoms.", style="yellow")
    else:
        # Generic error
        message.append(f"An unexpected error occurred: {exception}", style=style)
        if is_debug:
             # Show full traceback in debug mode
             console.print_exception(show_locals=True)
        else:
             message.append("\nSuggestion: Run with debug=True for a full traceback.", style="yellow")

    console.print(Panel(message, title=title, border_style="red"))

def display_warning(message: str, title: str = "Warning", suggestions: list = None) -> None:
    """Display a warning message using Rich."""
    text = Text(message, style="yellow")
    
    if suggestions:
        text.append("\n\nSuggestions:", style="bold yellow")
        for suggestion in suggestions:
            text.append(f"\n  - {suggestion}")
    
    console.print(Panel(text, title=title, border_style="yellow"))

def display_overwrite_warning(exception: FileOperationError) -> None:
    """Display a specific message for FileOperationError when overwrite is False."""
    message = Text()
    title = "Cautious Mode - File Exists"
    style = "bold red"
    message.append(f"Error: {exception.message}\n", style=style)
    if exception.src:
        message.append(f"  Source:      {exception.src}\n")
    if exception.dst:
        message.append(f"  Destination: {exception.dst}\n")
    message.append("\nYou are running in cautious mode (overwrite=False), so existing files were not overwritten.\n", style="yellow")
    message.append("To overwrite existing files, set overwrite=True.", style="bold green")
    console.print(Panel(message, title=title, border_style="red"))

def display_success(message: str, title: str = "Success") -> None:
    """Display a success message using Rich."""
    console.print(Panel(Text(message, style="green"), title=title, border_style="green")) 
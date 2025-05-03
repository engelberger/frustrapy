# Define custom exceptions for FrustraPy
class FrustraPyError(Exception):
    """Base exception for all frustrapy errors."""
    pass

class ValidationError(FrustraPyError):
    """Error for parameter validation failures."""
    def __init__(self, message: str, param_name: str = None, value: object = None):
        self.message = message
        self.param_name = param_name
        self.value = value
        super().__init__(self.__str__())

    def __str__(self):
        if self.param_name is not None and self.value is not None:
            return f"{self.message} (parameter '{self.param_name}' = {self.value!r})"
        return self.message

class SubprocessError(FrustraPyError):
    """Error for subprocess failures."""
    def __init__(self, message: str, cmd: list = None, returncode: int = None, stdout: str = None, stderr: str = None):
        self.message = message
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(self.__str__())

    def __str__(self):
        parts = [self.message]
        if self.cmd:
            parts.append(f"Command: {self.cmd}")
        if self.returncode is not None:
            parts.append(f"Return code: {self.returncode}")
        if self.stderr:
            parts.append(f"Error output:\n{self.stderr}")
        return "\n".join(parts)

class FileOperationError(FrustraPyError):
    """Error for file operation failures."""
    def __init__(self, message: str, src: str = None, dst: str = None):
        self.message = message
        self.src = src
        self.dst = dst
        super().__init__(self.__str__())

    def __str__(self):
        parts = [self.message]
        if self.src:
            parts.append(f"Source: {self.src}")
        if self.dst:
            parts.append(f"Destination: {self.dst}")
        return "\n".join(parts)

class MissingBackboneAtomError(FrustraPyError):
    """Error raised when PDB file is missing required backbone atoms."""
    def __init__(self, message: str, missing_atoms: list = None):
        self.message = message
        self.missing_atoms = missing_atoms or []
        super().__init__(self.__str__())
    
    def __str__(self):
        if not self.missing_atoms:
            return self.message
            
        parts = [self.message]
        parts.append("Missing backbone atoms:")
        for item in self.missing_atoms:
            if isinstance(item, tuple) and len(item) == 3:
                residue, chain, atom = item
                parts.append(f"  - Residue {residue} (Chain {chain}): missing {atom} atom")
            else:
                parts.append(f"  - {item}")
        return "\n".join(parts) 
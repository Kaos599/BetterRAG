"""
Utility to check for code consistency and logical bugs across the BetterRAG codebase.
"""

import os
import sys
import inspect
import importlib
import pkgutil
import re
from pathlib import Path

def check_method_signatures(module_name):
    """
    Check for method signature mismatches in a module.
    
    Args:
        module_name (str): The name of the module to check
    """
    print(f"Checking method signatures in {module_name}...")
    
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Could not import {module_name}")
        return
    
    # Check for classes in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            print(f"Checking class: {name}")
            
            # Get method references from class
            method_names = set()
            method_signatures = {}
            
            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                if not method_name.startswith('_'):  # Skip private methods
                    method_names.add(method_name)
                    sig = inspect.signature(method)
                    method_signatures[method_name] = sig
            
            # Check for references to undefined methods
            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                if not method_name.startswith('_'):  # Skip private methods
                    source = inspect.getsource(method)
                    
                    # Look for references to self.method_name()
                    for potential_method in method_names:
                        if f"self.{potential_method}" in source and potential_method != method_name:
                            # Check if the method exists
                            if not hasattr(obj, potential_method):
                                print(f"  WARNING: Method {method_name} references non-existent method {potential_method}")
                            else:
                                # Check for potential parameter mismatches
                                # Look for method calls with parameters
                                pattern = rf"self\.{potential_method}\s*\((.*?)\)"
                                calls = re.findall(pattern, source)
                                
                                if calls:
                                    called_method = getattr(obj, potential_method)
                                    called_sig = method_signatures.get(potential_method)
                                    
                                    if called_sig:
                                        # Count required parameters (excluding self)
                                        required_params = 0
                                        for param in list(called_sig.parameters.values())[1:]:  # Skip self
                                            if param.default == inspect.Parameter.empty and param.kind not in (
                                                inspect.Parameter.VAR_POSITIONAL, 
                                                inspect.Parameter.VAR_KEYWORD
                                            ):
                                                required_params += 1
                                        
                                        for call in calls:
                                            # Simple check: count arguments
                                            args = [a.strip() for a in call.split(',') if a.strip()]
                                            if len(args) < required_params:
                                                print(f"  WARNING: Call to {potential_method} in {method_name} may be missing required parameters")
                                                print(f"    Call: self.{potential_method}({call})")
                                                print(f"    Signature: {called_sig}")

def check_module_consistency():
    """Check for consistency across modules."""
    print("Checking module consistency...")
    
    # Get all Python files in the app directory
    app_dir = Path(__file__).parent.parent
    all_files = []
    
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('_'):
                rel_path = os.path.relpath(os.path.join(root, file), app_dir.parent)
                module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                all_files.append(module_path)
    
    # Check each file
    for module_path in all_files:
        check_method_signatures(module_path)
    
    print("Checking for common logical bugs...")
    check_common_logical_bugs(app_dir)

def check_common_logical_bugs(app_dir):
    """
    Check for common logical bugs in the codebase.
    
    Args:
        app_dir (Path): The root directory of the application
    """
    # Check for inconsistent method names
    print("Checking for inconsistent method names...")
    
    # Common method name pairs that should be consistent
    method_pairs = [
        ('connect', 'disconnect'),
        ('open', 'close'),
        ('start', 'stop'),
        ('initialize', 'cleanup'),
        ('create', 'delete'),
        ('add', 'remove'),
        ('get', 'set'),
        ('load', 'save'),
    ]
    
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    for method1, method2 in method_pairs:
                        if f"def {method1}" in content and f"def {method2}" not in content:
                            # Check if the method is actually used
                            if re.search(rf"self\.{method1}\s*\(", content):
                                print(f"  WARNING: File {file_path} defines '{method1}' but not '{method2}'")

if __name__ == "__main__":
    check_module_consistency() 
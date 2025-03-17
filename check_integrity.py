#!/usr/bin/env python
"""
Run code integrity checks on the BetterRAG codebase.
"""

from app.utils.check_code_integrity import check_module_consistency

if __name__ == "__main__":
    print("Running code integrity checks on BetterRAG codebase...")
    check_module_consistency()
    print("Integrity check complete.") 
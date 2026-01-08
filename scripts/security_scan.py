"""
Security Scanner - Detect sensitive data before committing to GitHub
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class SecurityScanner:
    def __init__(self, repo_root="."):
        self.repo_root = Path(repo_root)
        self.issues = []

        # Patterns to detect
        self.patterns = {
            'api_key': [
                r'api[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
                r'apikey\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
            ],
            'password': [
                r'password\s*[:=]\s*["\']([^"\']+)["\']',
                r'passwd\s*[:=]\s*["\']([^"\']+)["\']',
                r'pwd\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            'secret': [
                r'secret[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
                r'secret\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            'token': [
                r'token\s*[:=]\s*["\']([a-zA-Z0-9_\-]{20,})["\']',
                r'auth[_-]?token\s*[:=]\s*["\']([a-zA-Z0-9_\-]{20,})["\']',
            ],
            'aws_key': [
                r'AKIA[0-9A-Z]{16}',
                r'aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*["\']?([A-Z0-9]{20})["\']?',
            ],
            'private_key': [
                r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'ip_address': [
                r'\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)(?:[0-9]{1,3}\.){2}[0-9]{1,3}\b',
            ],
            'credit_card': [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
            ],
        }

        # Directories to skip
        self.skip_dirs = {
            '.git', 'venv', 'env', '.venv', 'node_modules',
            '__pycache__', '.pytest_cache', 'results', 'data/processed'
        }

        # Files to skip
        self.skip_files = {
            '.gitignore', '.DS_Store', 'security_scan.py'
        }

        # Extensions to scan
        self.scan_extensions = {
            '.py', '.js', '.json', '.yaml', '.yml', '.txt',
            '.md', '.sh', '.env', '.cfg', '.conf', '.ini'
        }

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for sensitive patterns"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for issue_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1

                        # Get the matched text (first 50 chars)
                        matched_text = match.group(0)[:50]

                        issues.append({
                            'file': str(file_path),
                            'line': line_num,
                            'type': issue_type,
                            'pattern': pattern,
                            'match': matched_text,
                            'severity': self.get_severity(issue_type)
                        })

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        return issues

    def get_severity(self, issue_type: str) -> str:
        """Get severity level for issue type"""
        high_severity = {'api_key', 'password', 'secret', 'token', 'aws_key', 'private_key', 'credit_card'}
        medium_severity = {'email', 'ip_address'}

        if issue_type in high_severity:
            return 'HIGH'
        elif issue_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'

    def check_large_files(self) -> List[Dict]:
        """Check for large files that shouldn't be committed"""
        large_files = []
        max_size = 50 * 1024 * 1024  # 50MB

        for file_path in self.repo_root.rglob('*'):
            if not file_path.is_file():
                continue

            # Skip excluded directories
            if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                continue

            if file_path.stat().st_size > max_size:
                large_files.append({
                    'file': str(file_path),
                    'size': file_path.stat().st_size / (1024 * 1024),  # MB
                    'severity': 'HIGH',
                    'type': 'large_file'
                })

        return large_files

    def check_sensitive_filenames(self) -> List[Dict]:
        """Check for files with sensitive names"""
        sensitive_names = [
            '.env', 'secret', 'password', 'credentials',
            'private', '.pem', '.key', 'id_rsa', 'config.json'
        ]

        sensitive_files = []

        for file_path in self.repo_root.rglob('*'):
            if not file_path.is_file():
                continue

            # Skip excluded directories
            if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                continue

            filename_lower = file_path.name.lower()
            for sensitive in sensitive_names:
                if sensitive in filename_lower:
                    sensitive_files.append({
                        'file': str(file_path),
                        'type': 'sensitive_filename',
                        'severity': 'MEDIUM',
                        'reason': f'Filename contains "{sensitive}"'
                    })
                    break

        return sensitive_files

    def scan_repository(self) -> Dict:
        """Scan entire repository"""
        print("=" * 60)
        print("Security Scan - Checking for sensitive data")
        print("=" * 60)
        print()

        all_issues = []

        # Scan files for patterns
        print("Scanning files for sensitive patterns...")
        scanned_count = 0

        for file_path in self.repo_root.rglob('*'):
            if not file_path.is_file():
                continue

            # Skip excluded directories
            if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                continue

            # Skip excluded files
            if file_path.name in self.skip_files:
                continue

            # Only scan specific extensions
            if file_path.suffix not in self.scan_extensions:
                continue

            scanned_count += 1
            issues = self.scan_file(file_path)
            all_issues.extend(issues)

        print(f"  Scanned {scanned_count} files")

        # Check for large files
        print("\nChecking for large files...")
        large_files = self.check_large_files()
        all_issues.extend(large_files)

        # Check for sensitive filenames
        print("Checking for sensitive filenames...")
        sensitive_files = self.check_sensitive_filenames()
        all_issues.extend(sensitive_files)

        return {
            'total_files_scanned': scanned_count,
            'issues': all_issues
        }

    def print_report(self, results: Dict):
        """Print scan results"""
        print("\n" + "=" * 60)
        print("Scan Results")
        print("=" * 60)

        issues = results['issues']

        if not issues:
            print("\n✅ No sensitive data detected!")
            print("   Repository appears safe to commit.")
            return True

        # Group by severity
        high = [i for i in issues if i['severity'] == 'HIGH']
        medium = [i for i in issues if i['severity'] == 'MEDIUM']
        low = [i for i in issues if i['severity'] == 'LOW']

        print(f"\n⚠️  Found {len(issues)} potential issues:")
        print(f"   🔴 HIGH:   {len(high)}")
        print(f"   🟡 MEDIUM: {len(medium)}")
        print(f"   🟢 LOW:    {len(low)}")

        # Print HIGH severity issues first
        if high:
            print("\n" + "🔴" * 20)
            print("HIGH SEVERITY ISSUES - DO NOT COMMIT!")
            print("🔴" * 20)

            for issue in high:
                print(f"\n  File: {issue['file']}")
                if 'line' in issue:
                    print(f"  Line: {issue['line']}")
                print(f"  Type: {issue['type']}")
                if 'match' in issue:
                    print(f"  Found: {issue['match']}")
                if 'reason' in issue:
                    print(f"  Reason: {issue['reason']}")
                if 'size' in issue:
                    print(f"  Size: {issue['size']:.2f} MB")

        # Print MEDIUM severity
        if medium:
            print("\n" + "🟡" * 20)
            print("MEDIUM SEVERITY - Review Before Committing")
            print("🟡" * 20)

            for issue in medium[:10]:  # Show first 10
                print(f"\n  File: {issue['file']}")
                if 'line' in issue:
                    print(f"  Line: {issue['line']}")
                print(f"  Type: {issue['type']}")
                if 'match' in issue:
                    print(f"  Found: {issue['match']}")
                if 'reason' in issue:
                    print(f"  Reason: {issue['reason']}")

            if len(medium) > 10:
                print(f"\n  ... and {len(medium) - 10} more")

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)

        if high:
            print("\n❌ DO NOT COMMIT - Fix high severity issues first!")
            print("\n1. Remove or sanitize sensitive data")
            print("2. Add sensitive files to .gitignore")
            print("3. Use environment variables for secrets")
            print("4. Re-run scan after fixes")
        elif medium:
            print("\n⚠️  Review medium severity issues carefully")
            print("\n1. Verify if data is truly sensitive")
            print("2. Sanitize if needed")
            print("3. Add to .gitignore if appropriate")

        return len(high) == 0


def main():
    scanner = SecurityScanner()
    results = scanner.scan_repository()
    safe = scanner.print_report(results)

    print("\n" + "=" * 60)
    if safe:
        print("✅ Repository scan complete - Safe to commit")
        sys.exit(0)
    else:
        print("❌ Security issues found - DO NOT commit yet")
        print("   See SECURITY.md for remediation steps")
        sys.exit(1)


if __name__ == "__main__":
    main()

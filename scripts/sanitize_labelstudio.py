"""
Sanitize Label Studio Export Files
Removes personal names and other PII from Label Studio metadata
"""

import json
import re
from pathlib import Path
import argparse


class LabelStudioSanitizer:
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)

    def sanitize_notes_json(self, output_path=None):
        """Sanitize notes.json file by removing personal names"""
        notes_file = self.input_dir / "notes.json"

        if not notes_file.exists():
            print(f"notes.json not found in {self.input_dir}")
            return

        with open(notes_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter categories to keep only actual classes
        if 'categories' in data:
            # Keep only ride and cowtail (or actual detection classes)
            actual_classes = []
            for cat in data['categories']:
                cat_name = cat.get('name', '').lower()

                # Keep if it's an actual detection class
                if cat_name in ['ride', 'cowtail', 'placeholder1', 'placeholder2',
                               'placeholder3', 'placeholder4', 'placeholder5',
                               'placeholder6', 'placeholder7']:
                    actual_classes.append(cat)

            # Remap IDs to be sequential
            for idx, cat in enumerate(actual_classes):
                cat['id'] = idx

            data['categories'] = actual_classes

        # Remove contributor info if present
        if 'info' in data and 'contributor' in data['info']:
            data['info']['contributor'] = 'Sanitized'

        # Save sanitized version
        if output_path is None:
            output_path = self.input_dir / "notes_sanitized.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Sanitized notes.json saved to: {output_path}")
        return output_path

    def sanitize_classes_txt(self, output_path=None):
        """Sanitize classes.txt file by removing personal names"""
        classes_file = self.input_dir / "classes.txt"

        if not classes_file.exists():
            print(f"classes.txt not found in {self.input_dir}")
            return

        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter to keep only actual detection classes
        sanitized_lines = []
        for line in lines:
            line_clean = line.strip().lower()

            # Keep only actual detection classes
            if line_clean in ['ride', 'cowtail'] or 'placeholder' in line_clean:
                sanitized_lines.append(line)

        # Save sanitized version
        if output_path is None:
            output_path = self.input_dir / "classes_sanitized.txt"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(sanitized_lines)

        print(f"✓ Sanitized classes.txt saved to: {output_path}")
        return output_path

    def create_safe_metadata(self, output_dir=None):
        """Create safe metadata files that can be committed to git"""
        if output_dir is None:
            output_dir = self.input_dir / "safe"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create safe notes.json
        safe_notes = {
            "categories": [
                {"id": 0, "name": "ride"},
                {"id": 1, "name": "cowtail"}
            ],
            "info": {
                "year": 2026,
                "version": "1.0",
                "contributor": "Project Team"
            }
        }

        notes_output = output_dir / "notes.json"
        with open(notes_output, 'w', encoding='utf-8') as f:
            json.dump(safe_notes, f, indent=2)

        print(f"✓ Created safe notes.json: {notes_output}")

        # Create safe classes.txt
        safe_classes = ["ride\n", "cowtail\n"]

        classes_output = output_dir / "classes.txt"
        with open(classes_output, 'w', encoding='utf-8') as f:
            f.writelines(safe_classes)

        print(f"✓ Created safe classes.txt: {classes_output}")

        # Create README
        readme_content = """# Safe Metadata Files

These files contain only class information without any personal data.
Safe to commit to public repositories.

## Files

- `notes.json` - Class definitions (sanitized)
- `classes.txt` - Class names list (sanitized)

## Original Data

Original Label Studio export files (with personal info) are in parent directory
and are excluded from git via .gitignore.
"""

        readme_output = output_dir / "README.md"
        with open(readme_output, 'w') as f:
            f.write(readme_content)

        print(f"✓ Created README: {readme_output}")

        return output_dir

    def scan_for_pii(self):
        """Scan files for potential PII"""
        print("\n" + "=" * 60)
        print("Scanning for PII in Label Studio files")
        print("=" * 60)

        pii_found = []

        # Check notes.json
        notes_file = self.input_dir / "notes.json"
        if notes_file.exists():
            with open(notes_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for email addresses
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                pii_found.append(('notes.json', 'Email addresses', emails))

            # Check for Korean names (common pattern: name + 님 + something)
            korean_names = re.findall(r'[\u3131-\uD79D]+님', content)
            if korean_names:
                pii_found.append(('notes.json', 'Korean names', korean_names))

            # Check for "checked" markers with names
            checked_markers = re.findall(r'[\u3131-\uD79D]+검사필', content)
            if checked_markers:
                pii_found.append(('notes.json', 'Name markers', checked_markers))

        # Check classes.txt
        classes_file = self.input_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                content = f.read()

            korean_names = re.findall(r'[\u3131-\uD79D]+님', content)
            if korean_names:
                pii_found.append(('classes.txt', 'Korean names', korean_names))

            checked_markers = re.findall(r'[\u3131-\uD79D]+검사필', content)
            if checked_markers:
                pii_found.append(('classes.txt', 'Name markers', checked_markers))

        # Report findings
        if pii_found:
            print("\n⚠️  PII DETECTED:")
            for filename, pii_type, items in pii_found:
                print(f"\n  File: {filename}")
                print(f"  Type: {pii_type}")
                print(f"  Count: {len(items)}")
                print(f"  Examples: {items[:3]}")
        else:
            print("\n✓ No obvious PII detected")

        return pii_found


def main():
    parser = argparse.ArgumentParser(
        description='Sanitize Label Studio export files to remove PII'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/project-8-at-2026-01-07-07-09-0780865d',
        help='Input directory containing Label Studio export'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for sanitized files'
    )
    parser.add_argument(
        '--scan-only',
        action='store_true',
        help='Only scan for PII, do not create sanitized files'
    )

    args = parser.parse_args()

    sanitizer = LabelStudioSanitizer(args.input)

    # Scan for PII
    pii_found = sanitizer.scan_for_pii()

    if args.scan_only:
        if pii_found:
            print("\n⚠️  PII found - files should NOT be committed to public repo")
            print("   Run without --scan-only to create sanitized versions")
            return 1
        else:
            print("\n✓ No PII found - files appear safe")
            return 0

    # Create sanitized versions
    if pii_found:
        print("\n" + "=" * 60)
        print("Creating Sanitized Files")
        print("=" * 60)

        output_dir = args.output if args.output else f"{args.input}/safe"

        sanitizer.create_safe_metadata(output_dir)

        print("\n" + "=" * 60)
        print("✅ Sanitization Complete!")
        print("=" * 60)
        print(f"\nSafe files created in: {output_dir}")
        print("\nThese files can be safely committed to a public repository.")
        print("Original files with PII remain in the source directory (excluded by .gitignore)")
    else:
        print("\n✓ No sanitization needed - no PII detected")

    return 0


if __name__ == "__main__":
    exit(main())

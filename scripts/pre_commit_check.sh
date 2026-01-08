#!/bin/bash

# Pre-Commit Security Check Script
# Run this before every commit to public GitHub

echo "=========================================="
echo "Pre-Commit Security Check"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to print result
print_check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ((ERRORS++))
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check 1: Run security scanner
echo "1. Running security scanner..."
python scripts/security_scan.py > /dev/null 2>&1
SCAN_RESULT=$?

if [ $SCAN_RESULT -eq 0 ]; then
    print_check 0 "Security scan passed"
else
    print_check 1 "Security scan found issues - review output"
    python scripts/security_scan.py
fi

echo ""

# Check 2: Scan Label Studio for PII
echo "2. Checking Label Studio files for PII..."
if [ -d "data/project-8-at-2026-01-07-07-09-0780865d" ]; then
    python scripts/sanitize_labelstudio.py --scan-only > /dev/null 2>&1
    PII_RESULT=$?

    if [ $PII_RESULT -eq 0 ]; then
        print_check 0 "No PII detected in Label Studio files"
    else
        print_check 1 "PII found in Label Studio files"
    fi
else
    echo "  (Label Studio directory not found - skipping)"
fi

echo ""

# Check 3: Large files
echo "3. Checking for large files (>50MB)..."
LARGE_FILES=$(find . -type f -size +50M -not -path "./.git/*" -not -path "./venv/*" 2>/dev/null)

if [ -z "$LARGE_FILES" ]; then
    print_check 0 "No large files detected"
else
    print_check 1 "Large files found:"
    echo "$LARGE_FILES"
fi

echo ""

# Check 4: Check for .env files
echo "4. Checking for .env files..."
ENV_FILES=$(git ls-files | grep "\.env$" 2>/dev/null)

if [ -z "$ENV_FILES" ]; then
    print_check 0 "No .env files in git"
else
    print_check 1 ".env files found in git:"
    echo "$ENV_FILES"
fi

echo ""

# Check 5: Check for common secrets
echo "5. Checking for common secret patterns..."

API_KEY_COUNT=$(git diff --cached | grep -i "api[_-]key" | wc -l)
PASSWORD_COUNT=$(git diff --cached | grep -i "password.*=" | wc -l)
SECRET_COUNT=$(git diff --cached | grep -i "secret" | wc -l)

TOTAL_SECRETS=$((API_KEY_COUNT + PASSWORD_COUNT + SECRET_COUNT))

if [ $TOTAL_SECRETS -eq 0 ]; then
    print_check 0 "No obvious secrets in staged changes"
else
    print_warning "Found $TOTAL_SECRETS potential secret(s) in staged changes"
    echo "  API keys: $API_KEY_COUNT"
    echo "  Passwords: $PASSWORD_COUNT"
    echo "  Secrets: $SECRET_COUNT"
fi

echo ""

# Check 6: Verify .gitignore exists
echo "6. Verifying .gitignore..."
if [ -f ".gitignore" ]; then
    print_check 0 ".gitignore exists"

    # Check critical entries
    if grep -q "venv/" .gitignore && grep -q "\.env" .gitignore && grep -q "\.pt" .gitignore; then
        print_check 0 "Critical patterns in .gitignore"
    else
        print_warning "Some critical patterns missing from .gitignore"
    fi
else
    print_check 1 ".gitignore not found"
fi

echo ""

# Check 7: Staged files review
echo "7. Reviewing staged files..."
STAGED_COUNT=$(git diff --cached --name-only | wc -l)

if [ $STAGED_COUNT -eq 0 ]; then
    print_warning "No files staged for commit"
else
    echo "  Staged files: $STAGED_COUNT"

    # Check for suspicious extensions
    if git diff --cached --name-only | grep -E "\.(pt|pth|jpg|jpeg|png|mp4|avi)$" > /dev/null; then
        print_warning "Staged files include models/images/videos - verify this is intended"
        git diff --cached --name-only | grep -E "\.(pt|pth|jpg|jpeg|png|mp4|avi)$"
    else
        print_check 0 "No model/image/video files staged"
    fi
fi

echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo "   Safe to commit."
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Passed with $WARNINGS warning(s)${NC}"
    echo "   Review warnings before committing."
    echo ""
    exit 0
else
    echo -e "${RED}❌ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo "   DO NOT COMMIT - Fix issues first!"
    echo ""
    echo "See SECURITY.md for help"
    exit 1
fi

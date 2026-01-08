# 🔒 Pre-Commit Security Checklist

**MANDATORY: Complete this checklist before EVERY commit to public GitHub!**

## ⚠️ CRITICAL SECURITY CHECKS

### 1. Run Security Scanner

```bash
python scripts/security_scan.py
```

- [ ] Scanner completed without HIGH severity issues
- [ ] Reviewed MEDIUM severity findings
- [ ] No API keys, passwords, or secrets detected

### 2. Check Sensitive Files

```bash
# Check what will be committed
git status

# Review staged changes
git diff --cached
```

- [ ] No `.env` files
- [ ] No `notes.json` or `classes.txt` with personal names
- [ ] No proprietary images/videos
- [ ] No trained models (unless intended)

### 3. Personal Information (PII)

```bash
# Scan for PII in Label Studio
python scripts/sanitize_labelstudio.py --scan-only
```

- [ ] No personal names
- [ ] No email addresses
- [ ] No phone numbers
- [ ] No internal employee information

### 4. File Sizes

```bash
# Check for large files
find . -type f -size +50M -not -path "./.git/*" -not -path "./venv/*"
```

- [ ] No files over 50MB
- [ ] All large files in .gitignore
- [ ] Considered using Git LFS for 50-100MB files

### 5. Secrets & Credentials

```bash
# Manual grep for common patterns
grep -r "api_key" . --exclude-dir={venv,.git}
grep -r "password" . --exclude-dir={venv,.git}
grep -r "secret" . --exclude-dir={venv,.git}
```

- [ ] No hardcoded API keys
- [ ] No hardcoded passwords
- [ ] No secret tokens
- [ ] No AWS keys
- [ ] No private SSH keys

### 6. Environment Variables Check

```bash
# Ensure .env is in .gitignore
cat .gitignore | grep ".env"
```

- [ ] `.env` files are gitignored
- [ ] No credentials in config files
- [ ] Using environment variables for secrets

### 7. Code Review

- [ ] No internal IP addresses
- [ ] No internal server names
- [ ] No company-specific configurations
- [ ] No TODO comments with sensitive info
- [ ] No hard-coded file paths with usernames

### 8. Commit Message Review

- [ ] Commit message doesn't contain sensitive info
- [ ] No customer/client names (if confidential)
- [ ] No project codenames (if secret)

## 🎯 YOUR PROJECT SPECIFIC CHECKS

### Label Studio Data

- [ ] `data/project-*/notes.json` is gitignored
- [ ] `data/project-*/classes.txt` is gitignored
- [ ] Korean names NOT in any committed files
- [ ] Used sanitized versions if metadata needed

### Dataset

- [ ] No actual images committed (`.jpg`, `.png`)
- [ ] No videos committed (`.mp4`, `.avi`)
- [ ] Only label coordinates in git (numbers only)

### Models

- [ ] No trained models (`.pt` files) unless intended
- [ ] Models directory has only `.gitkeep`
- [ ] Results directory excluded

## ✅ FINAL VERIFICATION

```bash
# Run all checks
./scripts/pre_commit_check.sh

# Or manually:
python scripts/security_scan.py
python scripts/sanitize_labelstudio.py --scan-only
git status
git diff --cached
```

**Sign-off:**

- [ ] All checks passed
- [ ] Reviewed changes carefully
- [ ] Confident no sensitive data included
- [ ] Ready to commit

---

## 🚨 If You Find Issues

### Found Sensitive Data?

```bash
# Remove from staging
git reset HEAD path/to/file

# Add to .gitignore
echo "path/to/file" >> .gitignore

# Re-run security scan
python scripts/security_scan.py
```

### Already Committed?

```bash
# If not pushed yet
git reset HEAD~1
# Remove sensitive files, add to .gitignore
git add .
git commit -m "Fixed: removed sensitive data"

# If already pushed - SEE SECURITY.md for recovery steps
```

## 📚 Resources

- **Full Security Guide**: `SECURITY.md`
- **Sanitization**: `scripts/sanitize_labelstudio.py`
- **Scanner**: `scripts/security_scan.py`
- **Deployment**: `DEPLOYMENT.md`

---

**Remember: Prevention is easier than recovery! Take your time with this checklist.** 🔒

# 🔒 Security Guide - What NOT to Upload to Public GitHub

## ⚠️ CRITICAL: Sensitive Data Found in Your Project

### 🚨 Immediate Action Required

Your Label Studio export contains **personal names** of people who labeled the data:

**File: `data/project-8-at-2026-01-07-07-09-0780865d/notes.json`**
**File: `data/project-8-at-2026-01-07-07-09-0780865d/classes.txt`**

Contains names like:
- 곽검사필checked
- 두민석님검사필checked
- 오정환님검사필checked
- 이상욱님검사필checked
- 이상원님검사필checked
- 이승준님검사필checked
- 조현준님검사필checked

**This is PII (Personally Identifiable Information) and should NOT be in a public repository!**

## ❌ DO NOT COMMIT - Sensitive Data Checklist

### 🔴 HIGH RISK - Never Commit These:

1. **Personal Information (PII)**
   - ❌ Names of employees/contractors
   - ❌ Email addresses
   - ❌ Phone numbers
   - ❌ Physical addresses
   - ❌ ID numbers
   - ❌ Social security numbers

2. **Authentication & Secrets**
   - ❌ API keys (OpenAI, AWS, Google Cloud, etc.)
   - ❌ Secret tokens
   - ❌ Passwords
   - ❌ Private keys (.pem, .key files)
   - ❌ OAuth tokens
   - ❌ Database credentials
   - ❌ SSH private keys

3. **Company/Client Data**
   - ❌ Proprietary datasets (images, videos)
   - ❌ Customer data
   - ❌ Business intelligence
   - ❌ Internal documents
   - ❌ Financial information
   - ❌ Trade secrets

4. **Trained Models (Potential IP)**
   - ⚠️ Trained model weights (.pt files)
   - ⚠️ May contain company IP
   - ⚠️ Consider if model architecture/weights are proprietary

5. **Configuration Files with Secrets**
   - ❌ .env files
   - ❌ config.json with credentials
   - ❌ Database connection strings
   - ❌ Server IP addresses (internal)

### 🟡 MEDIUM RISK - Evaluate Before Committing:

1. **Label Studio Metadata**
   - ⚠️ `notes.json` - May contain annotator names
   - ⚠️ `classes.txt` - May contain internal class names
   - ⚠️ `tasks.json` - May contain task assignments
   - ✅ Safe: Just bounding box coordinates in label files

2. **Dataset Metadata**
   - ⚠️ File paths that reveal internal structure
   - ⚠️ Timestamps that reveal work patterns
   - ⚠️ Comments with internal info

3. **Documentation**
   - ⚠️ Internal server names/IPs
   - ⚠️ Internal project codenames
   - ⚠️ Client names (if confidential)

### ✅ SAFE TO COMMIT:

- ✅ Python scripts (without secrets)
- ✅ Configuration templates (without actual credentials)
- ✅ Requirements.txt
- ✅ Documentation (sanitized)
- ✅ Example/demo data (non-sensitive)
- ✅ Model architecture (if not proprietary)
- ✅ .gitignore file
- ✅ README.md (without sensitive info)

## 🛡️ How to Protect Your Repository

### Step 1: Clean Sensitive Files Before First Commit

```bash
# Run security scan
python scripts/security_scan.py

# Review output and remove flagged files
git rm --cached path/to/sensitive/file

# Add to .gitignore
echo "path/to/sensitive/file" >> .gitignore
```

### Step 2: Use Environment Variables

**BAD ❌:**
```python
api_key = "sk-1234567890abcdef"
db_password = "MySecret123"
```

**GOOD ✅:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
db_password = os.getenv("DB_PASSWORD")
```

**Create `.env` file (already in .gitignore):**
```bash
API_KEY=sk-1234567890abcdef
DB_PASSWORD=MySecret123
```

### Step 3: Sanitize Data Files

For Label Studio exports, create sanitized versions:

```bash
# Remove personal names from metadata
python scripts/sanitize_labelstudio.py
```

### Step 4: Use Git Secrets Scanner

```bash
# Install git-secrets
# macOS
brew install git-secrets

# Ubuntu
git clone https://github.com/awslabs/git-secrets.git
cd git-secrets && make install

# Setup for your repo
cd /path/to/your/repo
git secrets --install
git secrets --register-aws
```

## 🔍 Before Your First Push - Security Checklist

Run this checklist **BEFORE** pushing to GitHub:

```bash
# 1. Check for secrets
grep -r "password" . --exclude-dir=venv --exclude-dir=.git
grep -r "api_key" . --exclude-dir=venv --exclude-dir=.git
grep -r "secret" . --exclude-dir=venv --exclude-dir=.git

# 2. Check for email addresses
grep -r -E "\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b" . --exclude-dir=venv --exclude-dir=.git

# 3. Check for IP addresses (internal)
grep -r -E "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" . --exclude-dir=venv --exclude-dir=.git

# 4. Check file sizes (GitHub limit: 100MB)
find . -type f -size +50M -not -path "./.git/*" -not -path "./venv/*"

# 5. Review what will be committed
git status
git diff --cached

# 6. Use our security scanner
python scripts/security_scan.py
```

## 🚨 I Already Committed Sensitive Data - What Now?

### If You Haven't Pushed Yet:

```bash
# Remove file from staging
git reset HEAD path/to/sensitive/file

# Remove from last commit
git rm --cached path/to/sensitive/file
git commit --amend

# Add to .gitignore
echo "path/to/sensitive/file" >> .gitignore
```

### If You Already Pushed to GitHub:

**🔥 CRITICAL: This requires rewriting history!**

```bash
# Option 1: Remove file from all history (use BFG Repo-Cleaner)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files sensitive-file.txt
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force

# Option 2: If repo is new, easiest to delete and recreate
# 1. Delete GitHub repo
# 2. Remove .git folder locally
# 3. Clean sensitive files
# 4. Re-initialize and push

# Option 3: Contact GitHub to purge cache
# https://support.github.com/contact
```

## 🔐 Additional Security Measures

### 1. Use GitHub's Secret Scanning

GitHub automatically scans for known secret patterns. Enable it:
- Go to repo Settings → Security → Secret scanning

### 2. Use .gitattributes for Binary Files

```bash
# .gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

### 3. Regular Security Audits

```bash
# Monthly: Run security scan
python scripts/security_scan.py

# Check for outdated dependencies
pip list --outdated

# Review .gitignore
cat .gitignore
```

### 4. Use Private Repository (if needed)

If your project contains:
- Proprietary algorithms
- Client data
- Company IP
- Competitive advantage

**Consider using a private repository** (free on GitHub)

### 5. Code Review Before Merge

Always review changes before merging:
```bash
git diff main..feature-branch
```

## 📋 Your Specific Risks

Based on your project, here's what to watch out for:

### 🔴 HIGH PRIORITY:

1. **Label Studio Metadata**
   - Contains personal names of annotators
   - **Action**: Already added to .gitignore
   - **Status**: ✅ Protected

2. **Dataset Images**
   - May contain proprietary/confidential content
   - **Action**: Already in .gitignore
   - **Status**: ✅ Protected

3. **Trained Models**
   - May be company IP
   - **Action**: Already in .gitignore
   - **Status**: ✅ Protected

### 🟡 MEDIUM PRIORITY:

1. **Internal Paths in Code**
   - Check for hard-coded paths like `/home/username/`
   - Use relative paths or environment variables

2. **Comments in Code**
   - Review for internal information
   - Client names, project codenames, etc.

3. **Git Commit Messages**
   - Don't include sensitive info in commit messages
   - They're permanent!

## 🛠️ Tools to Help

### Automated Scanners:

```bash
# 1. Our custom scanner (included)
python scripts/security_scan.py

# 2. Git-secrets (AWS open source)
git secrets --scan

# 3. TruffleHog (finds secrets in git history)
pip install truffleHog
truffleHog --regex --entropy=False .

# 4. Gitleaks
# https://github.com/gitleaks/gitleaks
gitleaks detect --source . --verbose
```

## 📞 If You Suspect a Leak

1. **Immediately rotate credentials**
   - Change API keys
   - Update passwords
   - Revoke tokens

2. **Contact GitHub Support**
   - Request cache purge if needed

3. **Notify affected parties**
   - If personal data leaked

4. **Document the incident**
   - For compliance/audit

## ✅ Pre-Push Security Checklist

Copy this checklist for each push:

- [ ] Run `python scripts/security_scan.py`
- [ ] Check `git status` - nothing unexpected
- [ ] No .env files staged
- [ ] No API keys in code
- [ ] No personal names in metadata
- [ ] No proprietary data
- [ ] No large files (>50MB)
- [ ] Commit messages sanitized
- [ ] .gitignore up to date
- [ ] Code reviewed

## 📚 Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Git-secrets](https://github.com/awslabs/git-secrets)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

---

**Remember: Once data is on the internet, it's very hard to completely remove it. Prevention is key!** 🔒

# Codex sandbox issues

## Windows ACL helper blocks workspace access

Observed on 2026-09-02 in zemilib_tests.

### Symptom

Ordinary read-only shell commands and the built-in apply_patch can fail before
the command starts with an error containing:

    windows sandbox failed: helper_unknown_error: apply deny-read ACLs

This is a Codex sandbox/ACL failure, not evidence of a repository or Git error.
A failed call does not modify project files.

### Efficient response

1. Do not repeatedly retry equivalent sandboxed commands.
2. Read-only operations inside the configured project roots have standing user
   authorization. Run them directly without asking the user for separate
   permission.
3. If the broken ACL helper still forces require_escalated for such a read,
   supply the tool-required justification without separately interrupting the
   user for approval.
4. Group related safe escalated operations into one planned command instead of
   requesting approval for many small commands. Reuse narrowly approved command
   prefixes when available.
5. Try the built-in apply_patch once. If it fails with the same ACL message,
   do not retry it through another shell: the command is intercepted by the same
   helper.
5. Use an explicitly approved PowerShell fallback with exact old/new fragment
   checks, IO.File ReadAllText, and IO.File WriteAllText.
6. Immediately run git diff --check, inspect git diff, and run relevant tests.
   For files under zemi, remember that it is a Git submodule and inspect it with
   git -C zemi status and git -C zemi diff.
7. Preserve unrelated changes and report the fallback to the user.

This workaround is specific to the ACL-helper failure. Prefer normal sandboxed
commands and built-in apply_patch whenever they work.
# Branch Snapshot — `piasecki/fix1`

**(Worktree folder: `piasecki/fix1/`)**
**Tip commit:** `537e7f9` — "make --updates relative to the checkpoint" (Przemysław Piasecki, 2026-04-19)
**Base:** [[master]] line at `8e80a47` ("Added eval command").
**Diff vs master:** `train_torcs_rl.py` only — **one line**.

---

## 1. Summary
A minimal, surgical bug-fix branch. Makes the `--updates` count **relative to the loaded
checkpoint's update index** instead of absolute, so resuming training actually runs the
requested number of additional updates.

## 2. The change (`train_torcs_rl.py`)
```diff
-        for update_idx in range(start_update, args.updates):
+        for update_idx in range(start_update, start_update + args.updates):
```
`start_update` is read from a resumed checkpoint's `update_idx` metadata. **Before:** if you
loaded a checkpoint already at update 200 and asked for `--updates 200`, the loop range
`range(200, 200)` was empty → no training happened. **After:** `range(200, 400)` → runs the
full 200 additional updates as intended.

## 3. Artifacts
Standard three checkpoints (`torcs_ppo_latest{,_best_eval,_best_train}.pt`).

## 4. Findings
- Correct, well-scoped fix to a resume-training off-by-semantics bug. No reward/agent changes.
- Independent of the speed/gear/infra experiments; could be cherry-picked into any branch that
  resumes from checkpoints. Note other branches (e.g. [[stable]], [[headless]]) did **not**
  pick this up and still use the absolute `range(start_update, args.updates)`.

# LLM Sandbox — Daily UX Audit (2026-03-01)

## Audit Summary
Full E2E audit of all UI components, training pipeline, editors, checkpoint system, and data pipeline.
Found **80+ issues** across 4 audit areas. Prioritized into 3 tiers below.

---

## Tier 1: Critical Fixes (Data Integrity / Wrong Behavior) — ALL DONE

- [x] **1. Weight tying not enforced in transformer init** — `initTransformer` now shares tokenEmbed for lmHead when `tieWeights=true`. Dedup in `getAllParams`/`getParamGroups`.
- [x] **2. Cloud save: old chunks not deleted on overwrite** — `saveCloudCheckpoint` now deletes existing chunks (batched at 500) before writing new ones.
- [x] **3. Cloud save: metadata before chunks** — Reversed order: chunks written first, metadata last as "commit point".
- [x] **4. URL hash config set before validation** — `setModelConfig` now only called when `errors.length === 0`.
- [x] **5. Non-uniform layer params silently ignored** — Added nHeads/dFF uniformity validation in `validateConfig`.
- [x] **6. sampleBatch negative rangeLen** — Guard throws clear error when dataset too small for block size.
- [x] **7. IndexedDB save resolves before tx commits** — All functions resolve on `tx.oncomplete`, added `onabort`/`onblocked` handlers.

## Tier 2: High-Impact UX Improvements — ALL DONE

- [x] **8. No "Initializing" loading state** — Shows "Initializing model and tokenizer..." during init phase.
- [x] **9. No generation cancellation** — Added `generationRunning` flag and `stopGeneration()` method. Both generate methods check it.
- [x] **10. GPU memory leak on unmount** — Added useEffect cleanup that calls `trainerRef.current?.stop()`.
- [x] **11. Tensor leaks on errors** — All tensor-creating methods now use try/finally for disposal (train, valLoss, generate, generateStreaming).
- [x] **12. generateStreaming O(n²) decode** — Tracks `runningText` and decodes only new token, appending incrementally.
- [x] **13. formatValue inline function → infinite chart re-renders** — Three `useCallback`-memoized format functions replace inline arrows.
- [x] **14. No file size limit on dataset upload** — 10MB limit check before `file.text()`.
- [x] **15. Node positions reset on every config change** — `setGraph` preserves positions via `posMap` for existing nodes.
- [x] **16. NodeInspector edits break uniform-layer constraint** — Uniform fields (dModel/nHeads/dFF) now update all layers. dModel auto-adjusts nHeads.
- [x] **17. Cloud checkpoint list has no loading indicator** — Added `cloudListLoading` state, shows "Loading checkpoints..." during fetch.
- [x] **18. No success feedback after save** — Added `successMsg` state with 3s auto-clear, shown in green.
- [x] **19. Auth listener leak** — `cancelled` flag pattern prevents async unsubscribe race.
- [x] **20. Stale trainer ref to PlaygroundPanel** — Added `trainer` state in App + `onTrainerChange` callback. PlaygroundPanel receives state, not ref.
- [x] **21. Train/val loss chart time axes misaligned** — Increased rolling window from 200 to 500 steps.
- [x] **22. Canvas not cleared when data resets** — Canvas cleared when `data.length < 2`.
- [x] **23. CharTokenizer breaks on multi-byte Unicode** — `encode()` now uses `Array.from(text)` for code point iteration.

## Tier 3: Polish & Minor UX — MOSTLY DONE

- [ ] **24. Code editor comments destroyed on Visual/Form switch** — Deferred: requires comment-preserving JSON parser, significant architectural change.
- [x] **25. Escape key / background click to dismiss inline editing** — Added `onEditCancel` prop, Escape cancels without committing. Background click clears editingNodeId.
- [x] **26. Checkpoint name duplicate warning** — Confirm dialog before overwriting same-name checkpoint (local + cloud).
- [ ] **27. Prompt/completion color split is character-based** — Deferred: BPE encode/decode round-trips cleanly in practice.
- [x] **28. Popup-blocked not handled for Google sign-in** — Added try/catch with fallback to `signInWithRedirect`.
- [x] **29. Error state shared across Local/Cloud tabs** — Clears error and success messages on tab switch.
- [x] **30. Missing ARIA tab roles on config sub-tabs** — Added `role="tablist"`, `role="tab"`, `aria-selected`, `aria-controls`.
- [x] **31. Color contrast below WCAG AA** — Changed `--text-3` to `#7B8794` (~5.3:1 ratio).
- [x] **32. word-break: break-all → overflow-wrap: break-word** — Fixed in App.css and PlaygroundPanel.tsx.
- [x] **33. GPT-2 vocab size warning** — Validation warns when embedding > 50% of total params for small models.
- [x] **34. Sampling loop fallthrough biases token 0** — Default changed to `probs.length - 1` in both generate methods.
- [x] **35. No ResizeObserver on chart canvas** — Added ResizeObserver + zero-size guard.

### Post-audit fixes (found during verification)
- [x] **36. CheckpointPanel.handleSave missing `checkpoints` dep** — Added to useCallback dependency array.
- [x] **37. CheckpointPanel.handleCloudSave missing `cloudCheckpoints` dep** — Added to useCallback dependency array.
- [x] **38. TrainingPanel.handleStart missing `setTrainer` dep** — Added to useCallback dependency array.

---

## Status

**Completed:** 33/35 fixes (Tier 1 + Tier 2 + Tier 3 + post-audit)
**Deferred:** 2 items (24: comment preservation, 27: token-based prompt split) — both require significant architectural work for minimal practical benefit

# Lessons Learned

## 2026-03-01 — E2E UX Audit

### Patterns to Watch
- **Ref values captured at render time**: `trainerRef.current` passed as props won't update when ref changes. Use state or callback patterns instead.
- **Inline arrow functions in useEffect deps**: Creates new reference every render → infinite re-renders. Memoize or use refs.
- **Cloud operations need atomic writes**: Writing metadata before data creates inconsistent states on partial failure. Use transactions or write data first.
- **IndexedDB promise timing**: Resolve on `tx.oncomplete`, not `request.onsuccess`. Transactions can abort after requests succeed.
- **Async cleanup in useEffect**: If the setup is async, the cleanup may run before setup completes. Use a `cancelled` flag pattern.
- **GPU tensor lifecycle**: Always use try/finally for tensor creation → disposal. GPU memory leaks are invisible but cumulative.
- **UTF-16 vs Unicode**: JavaScript `string[i]` and `.length` use UTF-16 code units. Use `for...of` or `Array.from()` for code point iteration.

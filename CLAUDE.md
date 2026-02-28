# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Sandbox** — a web-based platform for training small language models directly in the browser. Users can define, modify, and train transformer-based LLMs with real-time visualization of training progress and sample outputs.

### Core Goals
- Train small LLMs (nano-scale) entirely in-browser using WebGPU/WebGL compute
- Allow model architecture editing both graphically (visual node editor) and via code
- Show live training plots (loss curves, metrics) and sample text generation as training proceeds
- Ship with sample datasets following the nanoGPT/nanochat pattern (small chat-format corpora)

## Architecture

### Compute Layer
- In-browser tensor operations via WebGPU (preferred) with WebGL fallback
- Transformer building blocks: embedding, multi-head attention, FFN, layer norm, positional encoding
- Tokenizer (BPE or character-level) running client-side

### Model Builder
- **Code editor**: define architecture in a DSL or JS/TS config (layer types, dims, heads, etc.)
- **Visual editor**: drag-and-drop node graph where each node is a transformer component; changes sync bidirectionally with the code representation

### Training Engine
- Forward/backward pass + optimizer (Adam) running in web workers to keep UI responsive
- Streaming metrics (loss, perplance, tokens/sec) emitted via events
- Checkpoint save/load to IndexedDB or file export

### Visualization & Output
- Real-time loss/metric charts (e.g., using Chart.js, uPlot, or lightweight canvas)
- Live sample generation panel: periodically run inference during training and display generated text
- Architecture diagram auto-rendered from the current model config

### Data Pipeline
- Built-in sample datasets (tiny Shakespeare, small chat corpus) bundled or fetched on demand
- Dataset upload and preprocessing UI (tokenization, train/val split)

## Workflow Orchestration
### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution
### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project
### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it
### 6. Autonomous Bug Fizing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how
## Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections
## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

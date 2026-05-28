---
name: tdd-cycle
description: Implement → Verify → Refactor workflow for a discrete change. Use when working on a feature, bug fix, or focused chunk of work where the refactor step often gets skipped. Combines automated tests (preferred) with runtime verification; test-first and implementation-first are both fine. The mandatory part is the explicit refactor pass at the end.
---

# TDD Cycle

A three-step loop for working on one discrete change. The refactor step is the reason this skill exists — it's the part that gets skipped without a structured prompt, and it's where most code-quality value lives.

## When to use

- A discrete feature you're about to start.
- A bug fix where the fix should come with cleanup.
- An exploratory implementation that just landed and feels rough.
- Any time the user invokes `/tdd-cycle` or asks for a "cycle" on a change.

## When NOT to use

- Trivial changes (renaming a variable, fixing a typo, dependency bumps).
- Pure refactors with no behavior change — skip steps 1 and 2, just refactor.
- Spike or throwaway code.

## The three steps

### 1. Implement OR Test (either order is fine)

Pick whichever is easier and just start. Don't agonize:

- **Test-first** is usually easier when the behavior is easy to specify: pure functions, parsers, calculations, data transformations.
- **Implement-first** is usually easier when the behavior is visual, requires running the app, or has a fuzzy spec you'll clarify as you write.

If you wrote tests first, make sure they fail for the right reason before moving on. If you implemented first, you'll write the tests in step 2.

### 2. Verify

This step has **two parts**. Always attempt both:

**a. Automated test (preferred, but optional when impractical).**
- For Rust: `#[test]` or `#[cfg(test)]` modules, `cargo test`.
- For visual or interactive behavior (e.g. Bevy rendering, audio output, UI animations): an automated test usually does not exist that meaningfully verifies the change. Say so explicitly: *"No automated test fits — visual behavior."* Then move on.
- For mixed cases (game logic that's visible but also has pure data shape): write the test for the pure-data part and verify the visual part by running.

**b. Runtime verification (always).**
- `cargo build` to confirm the code compiles.
- `cargo run` (or equivalent) and exercise the path. State what you observed.
- If the change is visual and you can't observe it yourself (e.g. running inside a sandboxed agent), say so explicitly rather than claiming success.

If both verifications pass, proceed to step 3. If something fails, debug back at step 1.

### 3. Refactor (mandatory)

Do not skip this step, even when the diff looks clean. The previous steps produce code that *works*; this step produces code that's *good*.

Walk the diff (and the surrounding files it touches) looking for these specific things:

- **Duplication.** Same logic in 2+ places, or near-duplicates with small variations. Extract a helper or unify the shape.
- **Nested matches / conditionals.** `match` inside `match`, `if let` chains, `unreachable!()` arms. Usually a sign of a missing abstraction or an enum that should be split (e.g. `Action::Move(Direction)` instead of separate `North`/`South`/`East`/`West` variants on the top-level enum).
- **Magic numbers or strings** without a comment explaining where they came from or what they mean.
- **`unwrap()` / `expect()` / `unreachable!()` / `panic!()`** introduced in this change. Each one is a claim the path can't be hit — verify the claim, or refactor so it's true by construction.
- **Hardcoded values** that appear in more than one place. Lift to a `const`.
- **Comments explaining *what* the code does** rather than *why*. The code probably needs better names, not more comments.
- **Functions that grew long** in this change. Does a sub-step deserve its own function with a name that documents what it does?
- **Verbose ECS query filters / type signatures** that suggest a missing component, marker, or system-set abstraction.

**Output the refactor candidates explicitly** before applying any of them. Format:

> Refactor candidates:
> 1. *<observation>* — recommend / skip, reason
> 2. *<observation>* — recommend / skip, reason

Then apply the recommended ones. If you choose to skip a candidate, name the trade-off (e.g. "would add boilerplate for marginal gain"). Being explicit about skips is part of the discipline — it forces a real judgment instead of an implicit shrug.

After refactoring, **re-run step 2 verification** to make sure nothing broke. The refactor isn't done until the verification still passes.

## Output shape

When the cycle is complete, summarize:
- What was implemented (one sentence)
- How it was verified (test names + what was run + observed behavior)
- What was refactored (list of changes applied) and what was skipped (list with reasons)

This makes the cycle auditable — a future reviewer can see whether the refactor step was genuinely done or just rubber-stamped.

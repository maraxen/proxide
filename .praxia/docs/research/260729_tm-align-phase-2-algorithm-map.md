---
title: TM-align Phase 2 algorithm map
description: Line-referenced map of USalign's 5 seeding strategies, get_score_fast, DP_iter/NWDP_TM/TMscore8_search, and final multi-TM output
status: draft
task_id: 260729_tmalign_scaffold
date: '260729'
confidence: high
sources: '~/repos/USalign @ 177cc8a (v20240303), TMalign.h + NW.h + param_set.h'
---
# TM-align Phase 2 algorithm map

> **Addendum (260729, post-implementation review)**: implementing against this doc and then
> empirically parity-testing against the real `TMalign` binary surfaced 6 real deviations a
> dedicated Sonnet review found by re-reading the C++ source directly — this doc's summaries
> were mostly right but missed some exact semantics. Corrections, in case this doc is used again
> for a rewrite or a different port:
>
> - **`score_fun8`'s `score_sum_method=8`** (§2/§3 both use it) gates the *score sum* on a fixed
>   `score_d8` cutoff (`TMalign.h:11-51`: `if(di<=score_d8_cut) score_sum+=...`) — pairs beyond
>   `score_d8` contribute **zero**, not a small positive term. Unconditional summation over all
>   pairs is `score_sum_method=0`'s behavior, a different mode entirely. `score_d8 = 1.5*Lnorm^0.3
>   + 3.5` (already correctly captured in `d0.rs::score_d8`, this doc's §4 final-stage bullet on
>   `score_d8` was right — the miss was in not connecting it to `score_fun8`'s own gating).
> - **`get_score_fast`'s escalation** (§2): confirmed by direct read to be a genuine repeated
>   `while(1){...}` loop (`TMalign.h:550-571`, `:591-612`), not single-shot — this doc's original
>   phrasing ("escalating +0.5 if <3 pairs survive") was ambiguous on this point and the first
>   implementation pass read it as single-shot.
> - **`DP_iter`'s `t,u` are true input/output parameters** (§3): they persist across BOTH gap_open
>   passes within one `DP_iter` call, seeded before the call by the caller's own `detailed_search`
>   (a `TMscore8_search`-based refined fit, NOT a naive whole-alignment Kabsch fit) — not reset
>   per gap_open value. This doc's §3 said "t,u carry over to the next NWDP_TM call" but didn't
>   spell out that this persistence spans the *entire* `DP_iter` invocation, nor that the seeding
>   fit itself must come from `TMscore8_search`.
> - **Squared vs. unsquared distance**: `NW.h`'s rotation-aware `NWDP_TM` overload's `dist()` helper
>   (`basic_fun.h`) returns **squared** Euclidean distance, and `d02` is already squared
>   (`d01*d01`) — the score formula is `dist_sq/d02`, a dimensionless ratio. A port that takes
>   `sqrt()` of the distance before dividing by (squared) `d02` mixes units and shapes the entire
>   score matrix wrong. Worth stating explicitly since it's an easy mistake to repeat.
> - **`parameter_set4search`'s small-`Lnorm` branch differs from `parameter_set4final`'s**:
>   `param_set.h:18-19` uses `Lnorm<=19 → d0=0.168` for the search phase, vs. `parameter_set4final`
>   (`param_set.h:61`) using `Lnorm<=21 → d0=0.5`. Reusing one `base_d0` helper for both phases (as
>   the first implementation pass did) is wrong for small structures, even though it happens to be
>   latent/invisible for any test pair with `Lnorm` comfortably above both thresholds.
> - **Tie-breaks favor `>=`, not `>`**, in `get_initial`'s and `get_initial_fgt`'s best-candidate
>   selection (`TMalign.h:674`, `:1357`, `:1390`) — keeps the *last* candidate on an exact
>   floating-point tie. Only matters for symmetric/near-symmetric structure pairs.
>
> **`n_aligned` gap — closed (260729, second addendum).** The gap described in the paragraph below
> was **not** a DP/NWDP_TM convergence issue as originally hypothesized — it was a missing final-stage
> filter. `TMalign_main` (`TMalign.h:3554-3593`) walks the winning alignment under the *definitive*
> rotation and keeps only pairs within `score_d8` distance before reporting `Lali`/`n_ali8`; pairs
> that survived the raw DP but land far apart post-rotation don't count, even though `NWDP_TM`'s
> diagonal score (always positive — no penalty for a bad match) let them stay in the alignment map.
> `pipeline.rs`'s final stage never applied this filter, so `n_aligned` tracked the raw DP pair count
> instead of the reference's filtered `Lali`. Fixed by filtering the alignment by `score_d8` (using
> the already-correct `d0::score_d8` cutoff, previously only wired into `refine.rs`'s internal
> `score_fun8`) before computing both `n_aligned` and the final TM-score sum. Result on the same
> PDB1.pdb/PDB2.pdb pair: `n_aligned` now matches the reference **exactly** (119, was 163), and the
> TM-scores also tightened further (0.4265/0.6161 vs. reference 0.4265/0.6163 — within ~0.0002, down
> from ~0.004). Self-alignment remains an exact match (`n_aligned=250=250`, TM=1.0). Backlog #3788
> closed.
>
> **Still-open empirical gap (historical — see resolution above)**: even after fixing all 6 of the
> above, `crates/proxide-tmalign`'s `tmalign_pair_serial` on USalign's own bundled `PDB1.pdb`/`PDB2.pdb`
> sample (250×166 residues) produces `n_aligned=163` vs. the reference binary's `Lali=119`, despite
> TM-scores matching within ~0.004 absolute (0.4308/0.6205 vs. reference 0.4265/0.6163) both before
> and after the fixes — bit-identical output pre/post-fix on this specific pair, which is itself
> informative: `dp_iter` already iterates to `|ΔTM|<1e-6` convergence (up to 30×), so these were
> mostly *path*-level fixes that don't change *which fixed point* it converges to for this pair. The
> `n_aligned` gap is likely something more structural in the DP/NWDP_TM convergence dynamics — not yet
> root-caused, flagged as follow-up work for a dedicated debugging session (see backlog).

Ground-truth reference for implementing `crates/proxide-tmalign`'s Phase 2 (remaining seed
strategies + `DP_iter` refinement). Companion to
[specs/260729_proxide-tmalign-phases-2-5](../specs/260729_proxide-tmalign-phases-2-5.md), which
covers crate structure, parallelism, parity testing, and phase sequencing. This doc covers only
the algorithm internals — read it before writing any `seed/*.rs` or `refine.rs` code.

All line numbers are against `~/repos/USalign/TMalign.h` (and `NW.h`, `param_set.h`) at commit
`177cc8a`. Re-verify line numbers if the USalign clone has moved past that commit.

## 1. The 5 seeding strategies (`get_initial()` family)

**(a) Gapless threading — `get_initial()`, lines 642-691.** Slides sequence y over x with offset
`k` (diagonal shift), `k` ranging `n1=-ylen+min_ali` to `n2=xlen-min_ali` where
`min_ali=max(5, min(xlen,ylen)/2)`. For each `k`, builds map `y2x[j]=j+k` (or -1 if out of
`[0,xlen)`), scores it via `get_score_fast` (§2), keeps the best `k`. Step is 1 normally, 5 if
`fast_opt`. No Kabsch inside the loop except what `get_score_fast` does internally.

**(b) Secondary-structure alignment — `get_initial_ss()`, lines 928-933.** Runs
`NWDP_TM(path,val,secx,secy,xlen,ylen,gap_open=-1.0,y2x)` — the char-based NW overload (§3c) using
SS-letter identity as the score. Needs `secx`/`secy` from `make_sec()` (§5).

**(c) Local-structure superposition — `get_initial5()` (a.k.a. `initial5`), lines 943-1040.** Two
fragment lengths `n_frag = {20,100}` (capped at `aL/3` and `aL/2`, `aL=min(xlen,ylen)`). Jump steps
`n_jump1/n_jump2` depend on chain length: 45 (len>250), 35 (>200), 25 (>150), else 15, capped at
`len/3`; multiplied by 5 under `fast_opt`. For every `(i,j)` start pair on x and y with those
jumps, extracts an `n_frag`-length fragment, does `Kabsch` (fragment-only), then `NWDP_TM`
(rotation-aware overload, gap_open=0.0, `d02=(d0+1.5 clamped to D0_MIN)^2`) to get a full-length
map, scores it with `get_score_fast`, keeps best `GLmax`. Returns `false` (caller prints a warning)
if no candidate ever improves `GLmax=0`.

**(d) Local superposition + SS combo — `get_initial_ssplus()` (a.k.a. `initial3`), lines
1094-1104**, using helper `score_matrix_rmsd_sec()` (1042-1085): Kabsch-fits the *previous best
alignment* (`y2x0`, i.e. `invmap0` from seed (a)), rotates all of x, builds a full
`(xlen+1)x(ylen+1)` DP score matrix where
`score[i+1][j+1] = 1/(1+d_ij²/d02) + 0.5` if `secx[i]==secy[j]` else without the `+0.5` bonus
(`d02=(d0+1.5, clamped to D0_MIN)²`). Then `NWDP_TM(score,path,val,xlen,ylen,gap_open=-1.0,y2x)`
(score-matrix overload, §3a) produces the new map.

**(e) Fragment gapless threading — `get_initial_fgt()` (a.k.a. `initial4`), lines 1173-1402**,
using `find_max_frag()` (1107-1165). `find_max_frag` finds the longest contiguous run of CA-CA
distances `< dcu_cut` (starting `dcu0²=4.25²`, escalated by `dcu_cut=(1.1^inc * dcu0)²` until
fragment length ≥ `min(len/3, fra_min)`, `fra_min=4` (8 if `fast_opt`)) — done separately for x and
y giving `(xstart,xend)`/`(ystart,yend)`. The shorter of the two extracted fragments (`Lx` vs `Ly`;
ties broken by `xlen<ylen`) is then gapless-threaded against the full other sequence exactly like
seed (a), with `min_ali=max(fra_min-1, min_len/2.5)` and step 3 under `fast_opt` (or 1 exact).

Special-case in (e): if `Lx==Ly` *and* `xlen==ylen`, both directions are tried (lines 1207-1310,
asymmetry-avoidance for near-symmetric cases), and additionally if the extracted fragment equals
the *full* chain length (`L_fr==L0`), it's trimmed to the middle 79% (`n1=0.1*L0` to `n2=0.89*L0`,
lines 1220-1231, 1265-1277, 1314-1326) before threading. **Do not skip this trimming step** — it
changes which residues seed the alignment for short, high-identity structure pairs.

## 2. `get_score_fast()` — lines 488-633

Signature: `get_score_fast(r1,r2,xtm,ytm,x,y,xlen,ylen,invmap,d0,d0_search,t,u) -> double`. Given a
discrete map `invmap[j]=i`:

1. Gathers aligned pairs into `r1/r2` (=`xtm/ytm`), Kabsch-fits **once** over *all* aligned pairs,
   computes `tmscore = Σ 1/(1+d_i²/d0²)` (no d8 cutoff, not normalized — reference comment: "no
   need to normalize... will not be used for later scoring").
2. **Iteration 2**: recomputes a distance cutoff `d002t = max(d0_search², 3rd-smallest-squared-distance)`,
   filters pairs with `dist ≤ d002t` (escalating `+0.5` if `<3` pairs survive and `n_ali>3`),
   re-Kabsch-fits on the filtered subset, recomputes `tmscore1` over *all* original pairs using the
   new rotation.
3. **Iteration 3**: same again with `d002t = d0_search²+1` as the new base cutoff, giving `tmscore2`.

Returns `max(tmscore, tmscore1, tmscore2)`. This is the cheap scoring probe used by every
gapless-threading-style seed ((a)/(c)/(e)) to rank candidate offsets without a full
`TMscore8_search`.

## 3. DP core: `NWDP_TM()` (NW.h) and `DP_iter()` (TMalign.h:1408-1473)

Three `NWDP_TM` overloads share one DP structure (NW.h:1-435). Per the file's own header comment
(NW.h:1-11), it is a **simplified Gotoh** where gap-open==gap-extend, so no separate gap-state
matrices are needed — `path[i][j]` is a bool "came from diagonal" flag, and the
horizontal/vertical recurrences add `gap_open` only if the *previous* cell was itself diagonal
(`if(path[i-1][j]) h += gap_open`).

**This intentionally causes minor path asymmetry vs. a textbook Gotoh (per the reference comment)
but is ~1.5x faster, and must be replicated exactly bit-for-bit for parity — do not "fix" it into
a proper affine NW.**

- **(a)** `NWDP_TM(score, path, val, len1, len2, gap_open, j2i)` (NW.h:17-94): score matrix supplied
  precomputed (used by the ssplus seed).
- **(b)** `NWDP_TM(path, val, x, y, len1, len2, t, u, d02, gap_open, j2i)` (NW.h:99-180): score
  computed on the fly as
  `d[i][j] = val[i-1][j-1] + 1/(1+dist(rotate(t,u,x[i]), y[j])/d02)` — a "TM-score-weighted" cell
  value, not a distance penalty. Used by `get_initial5` and `DP_iter`.
- **(c)** char-based overload at NW.h:360 for SS-string alignment (used by `get_initial_ss`).

All three traceback identically: walk from `(len1,len2)`, follow `path[i][j]` if true (diagonal,
consumes both), else step in whichever of horizontal/vertical had ≥ value (ties broken toward
vertical: `if(v>=h) j--; else i--`).

`DP_iter()` is the outer refinement loop: `gap_open[2]={-0.6, 0}` — the caller passes a `(g1,g2)`
range selecting which gap_open value(s) to try (`(0,2)` = try both -0.6 and 0; the fgt seed passes
`(1,2)` = only gap_open=0). For each `g` in that range, runs up to `iteration_max` iterations (30
normal / 2 under `fast_opt`, but only 2 for the local-superposition seed regardless): each
iteration calls `NWDP_TM` (overload b) with current `t,u,d02=d0²` to get a new `invmap`, extracts
the aligned pairs into `xtm/ytm`, calls
`TMscore8_search(...,simplify_step=40,score_sum_method=8,...)` which internally re-Kabsch-fits and
returns a refined `tmscore` **and updates `t,u` in place** — so `t,u` carry over to the next
`NWDP_TM` call. Tracks `tmscore_max`/`invmap0` as running best across the whole double loop.
Converges/breaks early per `g` when `|tmscore - tmscore_old| < 1e-6` (checked only from
iteration>0 onward).

`TMscore8_search()` (lines 101-253) is itself a nested search: builds `n_init≤6` candidate
fragment lengths `L_ini = Lali/2^i` down to `L_ini_min=max(4,...)` (halving from full `Lali`), for
each fragment start position (stepped by `simplify_step`, capped at `iL_max=Lali-L_frag`) does a
local Kabsch fit, computes `score_fun8` (collects pairs with `dist<d` — with a fallback-widening
loop identical in spirit to `find_max_frag`'s: if fewer than 3 pairs survive and `n_ali>3`, grow
`d` by 0.5 repeatedly) at `d=local_d0_search-1`, then iteratively (`n_it=20` max) re-Kabsch-fits on
the surviving subset at `d=local_d0_search+1`, checking convergence via exact index-set equality
(`i_ali[k]==k_ali[k]` for all k) to break early. Global best `(t0,u0,score_max)` tracked across all
fragment lengths/positions — this is the routine that ultimately produces the `t,u` fed back into
`DP_iter`'s next `NWDP_TM` call.

## 4. Overall pipeline — `TMalign_main()` (TMalign.h:3138-3749)

Driven from `TMalign.cpp main()` around line 590. Parameters via `parameter_set4search()`
(param_set.h:9-28: `D0_MIN=0.5` → recomputed, `d0` from `Lnorm=min(xlen,ylen)` via
`d0=1.24*(Lnorm-15)^(1/3)-1.8` clamped `≥0.168`/`D0_MIN`; `d0_search=clamp(d0,4.5,8)`;
`score_d8=1.5*Lnorm^0.3+3.5`; `dcu0=4.25`). `ddcc = (Lnorm≤40) ? 0.1 : 0.4` gates whether a seed's
DP-iter refinement runs at all.

Sequence: gapless threading (a) → `detailed_search`/`TMscore8_search` → **DP_iter** (gap range
0..2, up to 30/2 iters) → SS seed (b) → detailed_search → `TM > TMmax*0.2` gate → DP_iter → local
superposition seed (c) → detailed_search → `TM > TMmax*ddcc` gate → DP_iter (2 iters only) → ssplus
seed (d) → detailed_search → `ddcc` gate → DP_iter → fgt seed (e) → detailed_search → `ddcc` gate
→ DP_iter (gap range 1..2, 2 iters).

**After every seed** the running-best `(TMmax, invmap0, t0/u0 if TMcut>0)` is updated — this is the
"best of all seeds" selection (simple `if(TM>TMmax)` overwrite; ties keep the earlier/first seed
since it's strict `>`). If `TMcut>0` (only used by `qTMclust`-style batch scanning, not plain
`TMalign` — **out of v1 scope**), after each stage `approx_TM()` early-exits against escalating
fractions of `TMcut`.

If `-i`/`-I` user-alignment options are set (`i_opt` 1-3), a `standard_TMscore` +
`detailed_search_standard` path runs instead of/alongside the built-in seeds — **out of v1 scope**
(v1 is algorithm-driven seeding only, no user-supplied alignment input).

**Final stage** (lines 3542-3749, unconditional once `invmap0` is non-empty): re-run
`detailed_search_standard(...,simplify_step=fast_opt?40:1,score_sum_method=8,bNormalize=false,...)`
on the final `invmap0` for the definitive `t,u`. Select pairs with `dist ≤ score_d8` (or all pairs
if `i_opt==3`) into `xtm/ytm`/`m1/m2`, re-Kabsch (`useWeight=0`) for final
`rmsd0=sqrt(rmsd/n_ali8)`. Then compute up to 5 differently-normalized TM-scores via
`parameter_set4final()` + `TMscore8_search` (**not** `get_score_fast`):

- `TM1` = normalized by `ylen` (structure 2 = "reference"; uses `t0,u0`) — **this is the canonical
  single TM-score** per `output_results`' printing order and the reference tool's own
  "(You should use TM-score normalized by length of the reference structure)" note.
- `TM2` = normalized by `xlen` (uses plain `t,u`, not `t0,u0`).
- `TM3` = avg-length normalized, only if `a_opt`.
- `TM4` = user-length normalized, only if `u_opt`.
- `TM5` = user-`d0` normalized (`parameter_set4scale`), only if `d_opt`.

Output finally contains: rotation `t0,u0` (used to write `-m` matrix file), `rmsd0`, `n_ali8`,
`Liden`, up to 5 `TM*`, and three aligned-sequence strings `seqxA/seqyA/seqM` reconstructed by
walking `m1/m2` pairs and filling gaps (lines 3670-3741) — `seqM[k]` is `:` if `dist<d0_out` else
`.` (`d0_out` defaults to 5.0, or `d0_scale` if `-d` used).

## 5. Other functions/data structures a faithful port needs

- **`make_sec(x,len,sec)`** (protein SS, TMalign.h:767-792) and **`sec_str(d13..d35)`** (737-762):
  pure-geometry SS assignment from 5 pairwise Cα-Cα distances against fixed helix/strand templates
  (`delta=2.1`/`1.42` tolerances) — no DSSP dependency; needed for seeds (b)/(d).
- **`smooth(sec,len)`** (693-735): post-processes the raw per-residue SS string (removes isolated
  single/double-residue runs, connects short gaps). **Verify this is actually invoked in the
  pipeline before porting** — no call site was found in `TMalign_main` during this exploration; it
  may be dead code for the plain-protein path, or used only by the RNA path / MMalign.
- **RNA SS** (`make_sec(seq,x,len,sec,atom_opt)` + `sec_str(len,seq,bp,a,b,c,d)`, 804-920):
  base-pair stack detection — **out of v1 scope** (locked-in decision: single-pair protein only).
- **`score_fun8`/`score_fun8_standard`** (13-99): the "collect pairs with `dist<d0`, sum TM
  contributions" primitive shared by `TMscore8_search`/`_standard`; `_standard` normalizes by
  `n_ali` instead of a caller-supplied `Lnorm`.
- **`detailed_search`/`detailed_search_standard`** (416-485): thin wrappers gathering `xtm/ytm`
  from an `invmap0` and calling `TMscore8_search`/`_standard`; `_standard` additionally rescales by
  `k/Lnorm` if `bNormalize`.
- **`standard_TMscore`** (3002-3070): used only for `-i`/`-I` and CP-align paths — **out of v1
  scope**.
- **`copy_t_u`** (3073-3081), **`approx_TM`** (3084-3115, fast TM estimate from a fixed rotation,
  `TMcut` early-exit only), **`clean_up_after_approx_TM`** (3117-3132) — low priority; `approx_TM`'s
  formula (`Σ1/(1+(d/d0)²)/Lnorm_0`, no d8 cutoff) only matters if batch-scan mode is ever ported.
- **`parameter_set4final_C3prime`/`parameter_set4final`/`parameter_set4scale`** (param_set.h) —
  `parameter_set4final` is already ported (`d0.rs`); confirm the RNA branch (irrelevant, v1 scope)
  and `parameter_set4scale`'s user-`d0` path (needed only if `TM5`/`-d` support is added) are
  understood as deferred, not missing.
- **`output_results`** (2831+) / **`output_rotation_matrix`** (2794+): exact printed numeric
  formats (`%7.5lf`/`%6.2f`) — relevant only if byte-identical CLI text output (not just numeric
  parity) becomes a goal; not needed for the `TmAlignResult` struct approach.

**Explicitly out of scope, do not conflate**: `CPalign_main` (circular permutation variant,
TMalign.h:3753+), `MMalign.h`/`SOIalign.h` (multi-chain / sequence-order-independent variants) —
separate algorithms in the same source tree, not part of plain `TMalign_main`.

# song_human_ai

Train a large sequence model to predict the next note of a musical theme —
the same underlying idea as a language model predicting the next token of
text, applied to full songs instead of drum patterns alone.

## Problem definition

A song is a sequence of discrete musical events (which instrument, which
pitch, when it starts, how long it lasts, how hard it's hit). If you flatten
a MIDI file into a sequence of such events, ordered in time, you get
something structurally identical to a token sequence in NLP: a long list of
discrete symbols with strong local and long-range dependencies (a melody
that returns, a chord progression that resolves, a chorus that repeats a
verse's rhythm at a different pitch).

The bet: train a **large autoregressive Transformer** on that event sequence
the way an LLM is trained on text — next-token prediction, causal
(left-to-right) attention, cross-entropy loss over a fixed vocabulary of
possible next events — and see whether it learns enough musical structure to
generate a **plausible continuation of a musical theme**, not just the next
plausible note in isolation.

This is a genuinely **generative** task: given a seed (a few bars, a motif,
or nothing at all), the model produces new notes that didn't exist before.
That's the key way this project differs from its predecessor,
`drum_bass_human` (a local, not-yet-published project; lessons from it are
captured in [`knowledge_drum_bass_human.md`](knowledge_drum_bass_human.md) in
this repo) — that project did per-hit *labelling* of an already-fixed
sequence of notes (predict how a human would nudge this exact note's timing
and velocity; pitch and rhythm were never touched, and the whole pattern was
visible at once via bidirectional attention). Here there is no fixed skeleton
to label — the model has to decide *what note comes next at all*, which is
why the architecture family changes from **encoder-only** to
**decoder-only / autoregressive**, mirroring the same shift that GPT-style
LLMs made over BERT-style ones for open-ended generation.

### What "predict the next note" actually means here

Concretely, this requires deciding on a **tokenization scheme** — how a note
(pitch, start time, duration, velocity, instrument) becomes one or more
discrete tokens the model can predict one at a time, analogous to a
tokenizer turning text into subword tokens. This is an open design question
this repo needs to resolve early (see Open questions below), likely along
the lines of established symbolic-music tokenizations (event-based schemes
like REMI/Compound-Word/MIDI-like, which represent time-shift, note-on,
note-off, velocity, and instrument as their own token types) rather than a
naive "one token per (pitch, time)" scheme, since the latter blows up
vocabulary size and loses the compositionality that makes token-based
modeling work in the first place.

## Dataset

- **~1,000 MIDI files** ("tabs"), each a full song transcription.
- **~3 minutes average length** per song.
- Unlike `drum_bass_human`'s drums-only, single-instrument-family corpus,
  these are presumably full arrangements (multiple instruments/tracks per
  song) — this needs to be confirmed by inspection before any architecture
  decision is finalized, since it directly determines the event vocabulary
  (how many distinct instruments/pitches exist) and the token sequence
  length per song (more simultaneous tracks -> more events per second of
  music -> longer sequences for the same wall-clock length).
- 1,000 songs is **small** by LLM-pretraining standards. This is closer to a
  low-resource language modeling setting than a "scale is all you need"
  one — data efficiency (augmentation, regularization, right-sized model
  capacity) will matter more here than in a data-rich regime. The prior
  project's own finding — that extra model *width* mostly bought overfitting
  once capacity stopped being the bottleneck, while extra *depth* kept
  helping at ~linear cost — is a concrete, already-tested precedent to lean
  on before assuming "bigger is better."

## Open questions (to resolve before training)

These are the design decisions this project needs to make explicitly, each
with a concrete, measurable way to answer it — following the "measure before
you decide" discipline from the prior project rather than guessing:

1. **Tokenization / vocabulary.** What exactly is one token? A pure
   event-based scheme (time-shift / note-on / note-off / velocity-bin /
   instrument, each its own token type) is the natural starting point;
   the actual vocabulary size should be *measured* from the real corpus
   (distinct pitches used, distinct instruments, velocity resolution
   needed), not assumed.
2. **Sequence length.** At ~3 minutes/song, how many tokens is that once
   tokenized? This must be **histogrammed from the real data** (median, p95,
   p99) before picking a training context window — `drum_bass_human`'s
   single costliest mistake was sizing a context window off a round number
   instead of the real distribution, and paying for it in wasted
   compute (see `knowledge_drum_bass_human.md`, §3). At full-song lengths,
   a single fixed context window may not even be the right call; sliding-
   window / chunked training (as `drum_bass_human` did per-section) is a
   likely necessity, not an optional optimization.
3. **Data quality.** Are all 1,000 tabs clean, correctly-transcribed,
   single-take performances? MIDI "tabs" sourced from hobbyist transcription
   sites are frequently quantized, incomplete, or mis-transcribed — the
   prior project's quality-filtering step (reject flat/degenerate examples
   before they teach the model the wrong thing) is directly applicable
   and should be budgeted for up front.
4. **Model size and capacity ladder.** With ~1k songs, what parameter count
   is actually supportable before overfitting dominates? This needs its own
   sweep, not an assumption carried over from the drum project (different
   data volume, different task).
5. **Evaluation.** "Next-token accuracy" / perplexity is the training-time
   proxy, but doesn't directly answer "does this sound musically coherent
   over a whole theme." A qualitative listening protocol (and ideally some
   quantitative structure-aware metric — e.g. does a generated section
   revisit earlier motifs at plausible intervals) needs to be defined
   before results can be judged, not improvised after the first model
   trains.

## Carrying lessons forward from `drum_bass_human`

[`knowledge_drum_bass_human.md`](knowledge_drum_bass_human.md) in this repo
is a full write-up of the architecture, data, and engineering lessons from
the prior project, written specifically to transfer to a *different* music-
modeling problem. The parts most relevant here going in:

- The **grid-search methodology** (screen cheap on a data subset, commit
  full compute only to the winning config, fingerprint what makes val
  numbers comparable across runs, persist the leaderboard as data not just
  stdout) applies unchanged to this project's own hyperparameter sweeps.
- The **checkpoint/resume gotchas** (a step-count-dependent LR schedule like
  `OneCycleLR` can't be naively resumed across a changed epoch/batch-size
  config; update "best" bookkeeping before writing the checkpoint that
  encodes it) apply to any Transformer training loop, including this one.
- The **run-naming collision bug** (formatting swept hyperparameters into a
  directory name with too little numeric precision silently merged two
  different runs) is a concrete reminder to test any config-to-string
  scheme for collisions before trusting a sweep's results.
- What does **not** carry over directly: the encoder-only, bidirectional,
  non-autoregressive architecture itself. This project's task is
  generation, not labelling, so the model family has to change — see
  "What 'predict the next note' actually means here" above.

## Status

Problem definition only. No code, cache pipeline, or model yet — next steps
are dataset inspection (answering the "Open questions" above with real
numbers from the 1,000 MIDI files) before any architecture or training code
is written.

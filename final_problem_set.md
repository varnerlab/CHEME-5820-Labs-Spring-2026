# Final Problem Set (PS13) Brainstorm

Scratchpad for the last problem set of the semester. Picks up from the L13d NanoGPT lab + L13a/L13c lecture content.

## The honest answer on "ask it questions and get decent answers"

**From scratch in a problem set: no.** Open-domain QA requires either pretraining at 10^8–10^10 token scale (GPT-2 small is the floor, and it's mediocre) or instruction-tuning on top of that. Neither fits in a 1–2 week problem set on student hardware.

**What actually fits and gives a satisfying "wow":**

- **TinyStories-style completion model.** Switch char → BPE, train ~10–30M params on the TinyStories dataset (~500MB), 2–4 hours on an M-series Mac. You get *coherent English* with real words, grammar, and short narrative arcs. Not QA, but visibly much better than char-Shakespeare.
- **Fine-tune a pretrained model for narrow QA.** Load GPT-2 small (or Phi-2) via `Transformers.jl`, fine-tune on a small domain Q&A corpus (could literally be the course material), and you *can* get "halfway decent" answers in a constrained domain.

## Killer PS13 outline (three tasks, tracking the lab structure)

1. **Tokenizer swap (char → BPE).** Keep the same NanoGPT architecture from L13d, swap the tokenizer, retrain on the same Shakespeare corpus. Compare perplexity, samples, and sequence-length efficiency. Pedagogical payoff: tokenization is the single biggest lever that doesn't involve retraining.

2. **Scale-up on TinyStories.** Same code, bigger `d`, `L`, `n_max`; train on TinyStories. Deliverable: coherent-English completions + a loss/perplexity plot that makes scaling laws visible. This is the "oh damn, it actually works" moment for students.

3. **Attention circuits: find the induction head.** Using the trained TinyStories model from Task 2, search across `(layer, head)` pairs for the *induction head* pattern (attends to the token after a previous match of the current token). This is the most-cited interpretability result in small transformers and connects directly to L13c.

## Main tradeoff

The above stays in Julia and stays pedagogically clean, but gives up "actually answer questions." If QA is the selling point, swap Task 3 for *"fine-tune GPT-2 on a course-content Q&A set via `Transformers.jl`"* — harder to grade, more moving parts, but ends with a demo the students can show their parents.

## Open decisions (pick up here)

- Purist scale-up-from-scratch version vs. fine-tune-a-pretrained-model version?
- Compute assumptions: what can we require of students (M-series Mac? GPU cluster access?)?
- Corpus for Task 2: TinyStories vs. something chemical-engineering-flavored?
- For the fine-tune variant: which base model (GPT-2 small, Phi-2, TinyLlama, Qwen-0.5B)?
- Deliverable format: notebook + generated samples + loss curves + write-up?

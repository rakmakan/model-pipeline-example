### Coding Challenge

This builds on the design document you submitted in stage 1 and the conversation we had about it. We'll use both as context when evaluating your submission, but you're not bound to either — if your thinking has evolved or something doesn't survive contact with the code, that's fine and worth calling out (in the README or the video).

We're providing a starter repo with: a synthetic dataset, a naive training script, a stub JSON model registry with one active model in it, and a naive promotion gate that compares accuracy only. The registry schema is intentionally minimal — extend it as you see fit. Treat the starter as the status quo to push back against, not a foundation to build on uncritically.

**Deliverables:**

1. **Training + registration pipeline.** Takes the versioned dataset, trains a model, evaluates it, and registers it as a candidate in the registry (version, training data hash, metrics, status). Documented commands are fine — no need for a single-command runner.
2. **Promotion gate.** Replace the naive gate with something you can defend. Include one or two tests covering the cases you think matter most. The gate is the piece we'll scrutinize most closely, so make the logic readable.
3. **A README** covering how to run the code, what you changed about the starter and why, how you'd monitor this specific model in production, and how you'd get a candidate from the registry to actually serving traffic. Be specific enough about *this* model and *this* code that we have something concrete to evaluate. Generic answers won't give us much to work with.
4. **A ~5 minute video walkthrough.** This is how you defend your work — there is no separate live interview for this stage. Walk us through the code, explain the choices behind it (especially the gate), and ideally include a live demo of the retrain → run → promotion flow end-to-end. Screen recording with voiceover is fine — production value is not the point. Be ready to navigate any code you submit, regardless of how it was written.

**What we're not asking for:**

- Don't build monitoring or deployment infrastructure. Write about it in the README and speak to it in the video.
- Don't build infrastructure for the rest either. Local files and JSON are fine.
- Don't aim for full test coverage on the gate. One or two well-chosen tests is the ask.

**Time:** Plan for 3 hours. Hard cap at 4. A smaller surface done well beats a sprawling unfinished thing.

**Deadline:** Submit within 1 week of receiving the repo.

**Logistics**

- **Getting the repo:** we'll send a zip. Python 3.10+; `pip install -r requirements.txt`. Start by reading the repo README.
- **Submitting:** either push your work to a **public GitHub repo** and email the link to `jon@hiretofu.com`, or **zip your repo** (including the `.git/` directory so we can see your commit history) and email it to `jon@hiretofu.com`. Either way, include your **video walkthrough** — a Loom / Google Drive / YouTube-unlisted link in the README is the easiest path. Include any notes you want us to read alongside it.
- **AI tools:** use whatever you'd use day-to-day. We don't care how the code got written; we care that you can defend every line of it in the video. Same applies to the README — if a section reads like generic LLM output, we'll notice.

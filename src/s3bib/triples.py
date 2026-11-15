"""Knowledge-graph triples: methods, data, theories, constructs, settings and findings.

A triple links a paper to what it uses or studies, or one construct to another:

    paper  USES_METHOD    "partial least squares"        (Method)
    paper  USES_DATA      "Community Innovation Survey"  (Dataset)
    paper  USES_THEORY    "resource-based view"          (Theory)
    paper  STUDIES        "absorptive capacity"          (Construct)
    paper  STUDIED_IN     "Germany"                      (Country)
    paper  IN_INDUSTRY    "manufacturing"                (Industry)
    paper  AT_LEVEL       "firm"                         (Level)
    "absorptive capacity"  INCREASES  "innovation performance"   (a finding)

Triples come from a file you upload (add_triples) or are extracted from titles and
abstracts by a language model (extract_triples):

    backend="claude-code"  your Claude plan, through the Claude Code command line (`claude -p`)
    backend="codex"        your ChatGPT plan, through the Codex command line (`codex exec`)
    backend="anthropic"    the Claude API (ANTHROPIC_API_KEY; pip install "s3-bibliometric[llm]")
    backend="openai"       the OpenAI API (OPENAI_API_KEY; pip install "s3-bibliometric[llm]")

In Claude Desktop or Claude Code, Claude can also do the extraction itself through the MCP
tools next_triple_batch and save_triples. Every triple keeps the sentence it rests on; triples
whose evidence is not found in the paper's title or abstract are flagged (evidence_found).
S3 joins the triples to the knowledge graph (s3bib.networks) and answers Q39 and Q40 with them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

ProgressCallback = Callable[[str, float], None]

PROMPT_VERSION = "s3-triples-1"
ENTITY_TYPES = ("Paper", "Method", "Dataset", "Theory", "Construct", "Country", "Industry", "Level")
# relation -> (subject type, object type); "Paper" subjects are the paper itself
PAPER_RELATIONS = {
    "USES_METHOD": "Method", "USES_DATA": "Dataset", "USES_THEORY": "Theory", "STUDIES": "Construct",
    "STUDIED_IN": "Country", "IN_INDUSTRY": "Industry", "AT_LEVEL": "Level",
}
FINDING_RELATIONS = ("INCREASES", "DECREASES", "NO_EFFECT_ON", "MODERATES", "MEDIATES", "ASSOCIATED_WITH")
RELATIONS = tuple(PAPER_RELATIONS) + FINDING_RELATIONS
_SYNONYMS = {
    "USES": "USES_METHOD", "METHOD": "USES_METHOD", "USES_DATASET": "USES_DATA", "DATA": "USES_DATA",
    "THEORY": "USES_THEORY", "STUDY": "STUDIES", "EXAMINES": "STUDIES", "COUNTRY": "STUDIED_IN",
    "INDUSTRY": "IN_INDUSTRY", "LEVEL": "AT_LEVEL", "POSITIVELY_AFFECTS": "INCREASES",
    "POSITIVE_EFFECT_ON": "INCREASES", "NEGATIVELY_AFFECTS": "DECREASES", "NEGATIVE_EFFECT_ON": "DECREASES",
    "NO_EFFECT": "NO_EFFECT_ON", "MODERATES_EFFECT": "MODERATES", "MEDIATES_EFFECT": "MEDIATES",
    "RELATED_TO": "ASSOCIATED_WITH", "ASSOCIATED": "ASSOCIATED_WITH",
}
COLUMNS = ["paper_id", "subject", "subject_type", "relation", "object", "object_type", "evidence",
           "evidence_found", "source"]

INSTRUCTIONS = """You extract a knowledge graph from research papers (title and abstract only).
For each paper, list what the paper itself reports, as triples:

Paper triples (subject "PAPER", subject_type "Paper"):
  USES_METHOD  -> Method     research methods, techniques, estimators, software (e.g. "fsQCA", "panel regression")
  USES_DATA    -> Dataset    named data sources, surveys, databases or samples (e.g. "Community Innovation Survey", "patent data")
  USES_THEORY  -> Theory     theories and frameworks the paper builds on (e.g. "resource-based view")
  STUDIES      -> Construct  the main constructs or variables studied (e.g. "absorptive capacity")
  STUDIED_IN   -> Country    countries or regions of the empirical setting
  IN_INDUSTRY  -> Industry   industries or sectors of the setting
  AT_LEVEL     -> Level      level of analysis: individual, team, firm, network, region or country

Finding triples (subject and object are constructs, both of type "Construct"):
  INCREASES, DECREASES, NO_EFFECT_ON, MODERATES, MEDIATES, ASSOCIATED_WITH
  e.g. "absorptive capacity" INCREASES "innovation performance".
  Use them only for relationships the abstract reports as findings or tests.

Rules:
- Only what the title or abstract states. Do not use outside knowledge, do not guess.
- Use short canonical names in lowercase, except proper names and acronyms ("CIS", "Germany").
- "evidence" is a short exact quote (up to 25 words) from the title or abstract that supports the triple.
- Use the paper's "id" exactly as given as "paper_id".
- A paper may have no triples of some kinds; return what is there.
- Answer with JSON only: {"triples": [{"paper_id": ..., "subject": ..., "subject_type": ...,
  "relation": ..., "object": ..., "object_type": ..., "evidence": ...}, ...]}"""

SCHEMA = {
    "type": "object",
    "properties": {"triples": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string"},
            "subject": {"type": "string"},
            "subject_type": {"type": "string", "enum": list(ENTITY_TYPES)},
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "object": {"type": "string"},
            "object_type": {"type": "string", "enum": list(ENTITY_TYPES)},
            "evidence": {"type": "string"},
        },
        "required": ["paper_id", "subject", "subject_type", "relation", "object", "object_type", "evidence"],
        "additionalProperties": False}}},
    "required": ["triples"],
    "additionalProperties": False,
}


class ExtractionError(RuntimeError):
    """The language model could not be reached or did not return triples."""


# --------------------------------------------------------------------------
# papers and batches
# --------------------------------------------------------------------------
def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "nan", "None")


def paper_key(row: pd.Series, i: int) -> str:
    """The id a paper has in triples: its EID, else its DOI, else its row number."""
    for col in ("eid", "doi"):
        if col in row and not _blank(row[col]):
            return str(row[col]).strip()
    return f"row{i}"


def paper_table(papers: pd.DataFrame) -> pd.DataFrame:
    """id, title and abstract for every paper (a run's papers or load_papers output)."""
    P = papers.reset_index(drop=True)
    return pd.DataFrame({"id": [paper_key(r, i) for i, r in P.iterrows()],
                         "title": P["title"].fillna("").astype(str),
                         "abstract": P["abstract"].fillna("").astype(str)})


def batches(table: pd.DataFrame, batch_size: int = 15) -> List[pd.DataFrame]:
    return [table.iloc[k:k + batch_size] for k in range(0, len(table), batch_size)]


def build_prompt(batch: pd.DataFrame) -> str:
    papers = [{"id": r.id, "title": r.title, "abstract": r.abstract} for r in batch.itertuples()]
    return INSTRUCTIONS + "\n\nPapers:\n" + json.dumps(papers, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------
# parsing and validation
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("‐", "-").replace("‑", "-")).strip().lower()


def _json_block(text: str) -> Any:
    """The JSON object in a model answer (also inside ```json fences or after prose)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise


def _relation(r: str) -> str:
    key = re.sub(r"[^A-Z_]", "", str(r).upper().replace(" ", "_").replace("-", "_"))
    return _SYNONYMS.get(key, key)


def _etype(t: str) -> str:
    t = str(t).strip().capitalize()
    return {"Data": "Dataset", "Theories": "Theory", "Variable": "Construct", "Concept": "Construct",
            "Sector": "Industry", "Region": "Country"}.get(t, t)


def validate(rows: Iterable[Dict[str, Any]], texts: Dict[str, str], source: str) -> pd.DataFrame:
    """Keep well-formed triples about the given papers; mark whether the evidence quote is in
    the paper's title or abstract."""
    out = []
    norm_text = {k: _norm(v) for k, v in texts.items()}
    for t in rows:
        pid = str(t.get("paper_id", "")).strip()
        if pid not in texts:
            continue
        rel = _relation(t.get("relation", ""))
        subj, obj = str(t.get("subject", "")).strip(), str(t.get("object", "")).strip()
        if rel not in RELATIONS or not obj or (rel in FINDING_RELATIONS and not subj):
            continue
        if rel in PAPER_RELATIONS:
            subj, st, ot = "PAPER", "Paper", PAPER_RELATIONS[rel]
        else:
            st, ot = "Construct", "Construct"
        ev = str(t.get("evidence", "") or "").strip()
        found = bool(ev) and _norm(ev).strip(" .\"'") in norm_text[pid]
        out.append({"paper_id": pid, "subject": subj, "subject_type": st, "relation": rel, "object": obj,
                    "object_type": ot, "evidence": ev, "evidence_found": found, "source": source})
    return pd.DataFrame(out, columns=COLUMNS)


def parse_response(text: str, batch: pd.DataFrame, source: str) -> pd.DataFrame:
    data = _json_block(text)
    rows = data.get("triples", []) if isinstance(data, dict) else data
    texts = {r.id: f"{r.title} {r.abstract}" for r in batch.itertuples()}
    return validate(rows if isinstance(rows, list) else [], texts, source)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def _cli(name: str, env_var: str) -> str:
    exe = os.environ.get(env_var) or shutil.which(name)
    if not exe:
        how = {"claude": "Install Claude Code (https://docs.claude.com/en/docs/claude-code) and sign in "
                         "with your Claude plan, or set S3BIB_CLAUDE to the claude executable.",
               "codex": "Install the Codex command line (npm install -g @openai/codex) and sign in with "
                        "your ChatGPT plan, or set S3BIB_CODEX to the codex executable."}[name]
        raise ExtractionError(f"The '{name}' command was not found. {how}")
    return exe


def _run_cli(cmd: List[str], prompt: str, timeout: float) -> str:
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ExtractionError(f"{Path(cmd[0]).name} did not answer within {timeout:.0f} s.")
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no message"]
        raise ExtractionError(f"{Path(cmd[0]).name} failed (exit {p.returncode}): {msg[0][:300]}")
    return p.stdout


def ask_claude_code(prompt: str, model: Optional[str] = None, timeout: float = 600) -> str:
    """One answer from Claude Code in print mode, using the user's Claude plan."""
    cmd = [_cli("claude", "S3BIB_CLAUDE"), "-p", "--output-format", "json"] + (["--model", model] if model else [])
    out = _run_cli(cmd, prompt, timeout)
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return out
    if isinstance(payload, dict):
        if payload.get("is_error"):
            raise ExtractionError(f"Claude Code: {str(payload.get('result', ''))[:300]}")
        return str(payload.get("result", ""))
    return out


def ask_codex(prompt: str, model: Optional[str] = None, timeout: float = 600) -> str:
    """One answer from the Codex command line (`codex exec`), using the user's ChatGPT plan.
    The prompt is read from standard input."""
    cmd = [_cli("codex", "S3BIB_CODEX"), "exec"] + (["--model", model] if model else []) + ["-"]
    return _run_cli(cmd, prompt, timeout)


def ask_anthropic(prompt: str, model: Optional[str] = None, timeout: float = 600) -> str:
    """One answer from the Claude API, constrained to the triples JSON schema.
    Credentials: ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    try:
        import anthropic
    except ImportError:
        raise ExtractionError('The Claude API needs the anthropic package: pip install "s3-bibliometric[llm]"')
    client = anthropic.Anthropic(timeout=timeout, max_retries=4)  # retries 429, 529 and other 5xx
    try:
        response = client.beta.messages.create(
            model=model or "claude-opus-5-5",
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",   # a declined request is retried on a fallback model
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise ExtractionError("The Claude API refused the key: set ANTHROPIC_API_KEY (never paste it in a chat).")
    except anthropic.RateLimitError:
        raise ExtractionError("The Claude API rate limit was reached after retries; try again later "
                              "or use a smaller batch_size.")
    except anthropic.APIStatusError as e:
        raise ExtractionError(f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise ExtractionError("Could not reach the Claude API: check the internet connection.")
    if response.stop_reason == "refusal":
        raise ExtractionError("Claude declined this batch.")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("The answer was cut off: use a smaller batch_size.")
    return next((b.text for b in response.content if b.type == "text"), "")


def ask_openai(prompt: str, model: Optional[str] = None, timeout: float = 600) -> str:
    """One answer from the OpenAI API in JSON mode. Credentials: OPENAI_API_KEY."""
    if not model:
        raise ExtractionError("Name the OpenAI model to use, e.g. model='<your model>'.")
    try:
        import openai
    except ImportError:
        raise ExtractionError('The OpenAI API needs the openai package: pip install "s3-bibliometric[llm]"')
    client = openai.OpenAI(timeout=timeout, max_retries=4)
    try:
        r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": prompt}])
    except openai.AuthenticationError:
        raise ExtractionError("The OpenAI API refused the key: set OPENAI_API_KEY (never paste it in a chat).")
    except openai.APIError as e:
        raise ExtractionError(f"OpenAI API error: {e}")
    return r.choices[0].message.content or ""


BACKENDS = {"claude-code": ask_claude_code, "codex": ask_codex, "anthropic": ask_anthropic, "openai": ask_openai}


def available_backends() -> Dict[str, str]:
    """Which extraction routes this computer can use now, and why not otherwise."""
    out = {}
    out["claude-code"] = "ready" if (os.environ.get("S3BIB_CLAUDE") or shutil.which("claude")) else \
        "install Claude Code and sign in with your Claude plan"
    out["codex"] = "ready" if (os.environ.get("S3BIB_CODEX") or shutil.which("codex")) else \
        "install the Codex command line and sign in with your ChatGPT plan"
    try:
        import anthropic  # noqa: F401
        out["anthropic"] = "ready" if os.environ.get("ANTHROPIC_API_KEY") else "set ANTHROPIC_API_KEY"
    except ImportError:
        out["anthropic"] = 'pip install "s3-bibliometric[llm]" and set ANTHROPIC_API_KEY'
    try:
        import openai  # noqa: F401
        out["openai"] = "ready" if os.environ.get("OPENAI_API_KEY") else "set OPENAI_API_KEY"
    except ImportError:
        out["openai"] = 'pip install "s3-bibliometric[llm]" and set OPENAI_API_KEY'
    return out


# --------------------------------------------------------------------------
# extraction with progress and a resumable cache
# --------------------------------------------------------------------------
def _eta(seconds: float) -> str:
    if seconds < 90:
        return f"about {max(1, round(seconds))} s left"
    return f"about {round(seconds / 60)} min left"


def extract_triples(papers: Union[pd.DataFrame, Any], backend: str = "claude-code", model: Optional[str] = None,
                    batch_size: int = 15, max_papers: Optional[int] = None,
                    cache_dir: Optional[str] = "~/s3bib_runs/_triples", timeout: float = 600,
                    progress: Optional[ProgressCallback] = None, stop_on_error: bool = False) -> pd.DataFrame:
    """Extract triples from titles and abstracts with a language model, batch by batch.

    papers: an S3 result, or a data frame with title and abstract (and EID/DOI if present).
    Each batch's answer is cached, so an interrupted run continues where it stopped and a
    second run costs nothing. progress(stage, fraction) reports e.g.
    "Triples: batch 3 of 40 (45 of 600 papers), about 4 min left"."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}")
    P = papers.papers if hasattr(papers, "papers") else papers
    table = paper_table(P)
    table = table[(table["title"].str.strip() != "") | (table["abstract"].str.strip() != "")]
    table = table.drop_duplicates("id")
    if max_papers:
        table = table.head(int(max_papers))
    parts = batches(table, batch_size)
    ask = BACKENDS[backend]
    cdir = Path(cache_dir).expanduser() if cache_dir else None
    say = progress or (lambda s, p: None)
    found, errors, t0, fresh = [], [], time.time(), 0
    done_papers = 0
    say(f"Triples: {len(table)} papers in {len(parts)} batches with {backend}", 0.0)
    for k, batch in enumerate(parts):
        prompt = build_prompt(batch)
        key = hashlib.sha1(f"{PROMPT_VERSION}|{backend}|{model}|{prompt}".encode()).hexdigest()
        f = cdir / f"{key}.json" if cdir else None
        try:
            if f is not None and f.exists():
                text = json.loads(f.read_text())["answer"]
            else:
                text = ask(prompt, model=model, timeout=timeout)
                fresh += 1
                if f is not None:
                    cdir.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps({"backend": backend, "model": model, "answer": text}))
            found.append(parse_response(text, batch, backend))
        except (ExtractionError, ValueError, json.JSONDecodeError) as e:
            if stop_on_error or isinstance(e, ExtractionError) and "not found" in str(e):
                raise
            errors.append(f"batch {k + 1}: {str(e)[:200]}")
        done_papers += len(batch)
        spent = time.time() - t0
        left = spent / max(fresh, 1) * sum(1 for j in range(k + 1, len(parts))) if fresh else 0
        say(f"Triples: batch {k + 1} of {len(parts)} ({done_papers} of {len(table)} papers)"
            + (f", {_eta(left)}" if fresh and k + 1 < len(parts) else ""), (k + 1) / len(parts))
    out = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=COLUMNS)
    out.attrs["errors"] = errors
    return out


# --------------------------------------------------------------------------
# loading triples and joining them to a run
# --------------------------------------------------------------------------
_ALIASES = {
    "paper_id": ["paper_id", "paper", "eid", "doi", "id", "title", "document"],
    "subject": ["subject", "head", "source_entity", "from"],
    "relation": ["relation", "predicate", "rel", "edge", "type"],
    "object": ["object", "tail", "target_entity", "to", "value"],
    "subject_type": ["subject_type", "head_type", "from_type"],
    "object_type": ["object_type", "tail_type", "to_type", "entity_type"],
    "evidence": ["evidence", "quote", "sentence", "support"],
    "source": ["source", "extractor", "model"],
}


def read_triples(data: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """A triples file (CSV, Excel, JSON or JSON lines) or table, with flexible column names."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        p = Path(data).expanduser()
        suf = p.suffix.lower()
        if suf in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif suf == ".jsonl":
            df = pd.read_json(p, lines=True)
        elif suf == ".json":
            raw = json.loads(p.read_text())
            df = pd.DataFrame(raw.get("triples", raw) if isinstance(raw, dict) else raw)
        else:
            df = pd.read_csv(p, sep=None, engine="python")
    lower = {c.strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for col, names in _ALIASES.items():
        hit = next((lower[n] for n in names if n in lower), None)
        out[col] = df[hit] if hit is not None else None
    missing = [c for c in ("paper_id", "relation", "object") if out[c].isna().all()]
    if missing:
        raise ValueError(f"The triples need columns for {', '.join(missing)} "
                         "(e.g. paper_id, subject, relation, object).")
    return out


def _canonical(names: pd.Series) -> pd.Series:
    """Merge spelling variants (case, spaces, hyphens, a trailing plural s): every variant gets
    the most frequent spelling."""
    key = names.map(lambda s: re.sub(r"[\s\-_]+", " ", str(s).strip().lower()).rstrip("s").strip())
    best = names.groupby(key).agg(lambda s: s.value_counts().index[0])
    return key.map(best)


def match_papers(triples: pd.DataFrame, papers: pd.DataFrame) -> Tuple[pd.Series, int]:
    """Row of the run's papers for each triple, by EID, DOI or title; -1 if not found."""
    from .pipeline import _norm_text
    P = papers.reset_index(drop=True)
    look: Dict[str, int] = {}
    for i, r in P.iterrows():
        for col in ("eid", "doi"):
            if col in P and not _blank(r[col]):
                look.setdefault(str(r[col]).strip().lower(), i)
        look.setdefault(f"row{i}", i)
        t = _norm_text(r["title"])
        if t:
            look.setdefault("title:" + t, i)
    def find(v):
        if _blank(v):
            return -1
        s = str(v).strip()
        return look.get(s.lower(), look.get("title:" + _norm_text(s), -1))
    idx = triples["paper_id"].map(find)
    return idx, int((idx < 0).sum())


def prepare_triples(triples: pd.DataFrame, papers: pd.DataFrame, source: str = "upload",
                    aliases: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean triples and match them to the run's papers. Returns (triples, report)."""
    t = read_triples(triples) if not set(COLUMNS[:6]).issubset(triples.columns) else triples.copy()
    t["relation"] = t["relation"].map(_relation)
    bad_rel = int((~t["relation"].isin(RELATIONS) & t["relation"].notna()).sum())
    unknown = sorted(set(t.loc[~t["relation"].isin(RELATIONS), "relation"].dropna()))[:10]
    t = t[t["relation"].isin(RELATIONS) & t["object"].notna()].copy()
    for col in ("subject", "object"):
        t[col] = t[col].fillna("").astype(str).str.strip()
    is_paper = t["relation"].isin(list(PAPER_RELATIONS))
    t.loc[is_paper, "subject"] = "PAPER"
    t.loc[is_paper, "subject_type"] = "Paper"
    t.loc[is_paper, "object_type"] = t.loc[is_paper, "relation"].map(PAPER_RELATIONS)
    t.loc[~is_paper, "subject_type"] = "Construct"
    t.loc[~is_paper, "object_type"] = "Construct"
    t = t[(t["object"] != "") & (is_paper | (t["subject"] != ""))]
    if aliases:
        amap = {k.strip().lower(): v for k, v in aliases.items()}
        for col in ("subject", "object"):
            t[col] = t[col].map(lambda s: amap.get(s.strip().lower(), s))
    # one spelling per entity, per type
    for typ in set(t["object_type"]):
        m = t["object_type"] == typ
        t.loc[m, "object"] = _canonical(t.loc[m, "object"])
    m = t["subject_type"] == "Construct"
    if m.any():
        both = pd.concat([t.loc[m, "subject"], t.loc[t["object_type"] == "Construct", "object"]])
        canon = dict(zip(both, _canonical(both)))
        t.loc[m, "subject"] = t.loc[m, "subject"].map(canon)
        mo = t["object_type"] == "Construct"
        t.loc[mo, "object"] = t.loc[mo, "object"].map(canon)
    idx, unmatched = match_papers(t, papers)
    t["paper"] = idx.astype(int)
    t = t[t["paper"] >= 0]
    if "source" not in t or t["source"].isna().all():
        t["source"] = source
    t["source"] = t["source"].fillna(source)
    if "evidence_found" not in t:
        t["evidence_found"] = pd.NA
    t = t.drop_duplicates(["paper", "subject", "relation", "object"]).reset_index(drop=True)
    report = {"triples": int(len(t)), "papers_with_triples": int(t["paper"].nunique()),
              "unmatched_triples": unmatched, "unknown_relations": bad_rel,
              "unknown_relation_names": unknown,
              "by_relation": t["relation"].value_counts().to_dict()}
    return t[["paper"] + COLUMNS], report


def add_triples(res, data, source: str = "upload", aliases: Optional[Dict[str, str]] = None,
                replace: bool = False) -> Dict[str, Any]:
    """Join triples (a file, a table or extract_triples output) to a run. They go into the
    knowledge graph and answer Q39 and Q40. Returns how many matched and what was left out."""
    t, report = prepare_triples(data if isinstance(data, pd.DataFrame) else read_triples(data),
                                res.papers, source, aliases)
    old = getattr(res, "triples", None)
    if old is not None and len(old) and not replace:
        t = pd.concat([old, t], ignore_index=True).drop_duplicates(["paper", "subject", "relation", "object"])
    res.triples = t.reset_index(drop=True)
    for k in ("_layers",):
        if hasattr(res, k):
            setattr(res, k, None)
    report["triples_in_run"] = int(len(res.triples))
    return report


def extract_and_add(res, backend: str = "claude-code", model: Optional[str] = None, batch_size: int = 15,
                    max_papers: Optional[int] = None, progress: Optional[ProgressCallback] = None,
                    **kw) -> Dict[str, Any]:
    """extract_triples on the run's papers, then add_triples. Returns the add_triples report
    plus the batches that failed."""
    t = extract_triples(res, backend, model, batch_size, max_papers, progress=progress, **kw)
    report = add_triples(res, t, source=backend)
    report["failed_batches"] = t.attrs.get("errors", [])
    report["evidence_found_share"] = round(float(t["evidence_found"].mean()), 3) if len(t) else None
    return report


# --------------------------------------------------------------------------
# the in-chat route (MCP): the assistant extracts, S3 hands out batches and keeps count
# --------------------------------------------------------------------------
def pending_papers(res) -> pd.DataFrame:
    table = paper_table(res.papers)
    table = table[(table["title"].str.strip() != "") | (table["abstract"].str.strip() != "")]
    done = set(getattr(res, "triples_done", set()) or set())
    return table[~table["id"].isin(done)]


def chat_progress(res) -> Dict[str, Any]:
    table = paper_table(res.papers)
    total = int(((table["title"].str.strip() != "") | (table["abstract"].str.strip() != "")).sum())
    done = len(set(getattr(res, "triples_done", set()) or set()) & set(table["id"]))
    return {"papers_done": done, "papers_total": total, "percent": round(100 * done / total, 1) if total else 100.0,
            "triples": int(len(getattr(res, "triples", None) if getattr(res, "triples", None) is not None else []))}


def next_batch(res, batch_size: int = 15) -> Dict[str, Any]:
    left = pending_papers(res)
    prog = chat_progress(res)
    if left.empty:
        return {"done": True, "progress": prog}
    b = left.head(batch_size)
    return {"done": False, "progress": prog, "instructions": INSTRUCTIONS,
            "papers": [{"id": r.id, "title": r.title, "abstract": r.abstract} for r in b.itertuples()]}


def save_chat_batch(res, triples: List[Dict[str, Any]], paper_ids: Optional[Sequence[str]] = None,
                    source: str = "claude-chat") -> Dict[str, Any]:
    """Validate the assistant's triples for one batch, add them, and mark the batch's papers done
    (also papers that had nothing to extract)."""
    table = paper_table(res.papers).set_index("id")
    ids = list(paper_ids) if paper_ids else sorted({str(t.get("paper_id", "")) for t in triples})
    ids = [i for i in ids if i in table.index]
    texts = {i: f"{table.at[i, 'title']} {table.at[i, 'abstract']}" for i in ids}
    v = validate(triples, texts, source)
    report = add_triples(res, v, source=source) if len(v) else {"triples": 0}
    done = set(getattr(res, "triples_done", set()) or set())
    res.triples_done = done | set(ids)
    prog = chat_progress(res)
    return {"saved": int(len(v)), "evidence_found": int(v["evidence_found"].sum()) if len(v) else 0,
            "progress": prog,
            "message": f"Saved {len(v)} triples. {prog['papers_done']} of {prog['papers_total']} papers done "
                       f"({prog['percent']:.0f}%)."}

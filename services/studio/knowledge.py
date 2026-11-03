"""Knowledge for Studio agents: document ingestion + retrieval.

This is retrieval-augmented prompting, NOT model fine-tuning: at call time the
caller's utterance is matched against the agent's Q&A pairs, script lines and
document chunks, and the best few are put in the prompt. Ranking is BM25 over
Unicode word tokens (works for Hindi/Gujarati/English with no model or extra
service, so it adds ~0 latency on the call path).
"""

import csv
import io
import ipaddress
import json
import math
import re
import socket
import zipfile
from collections import Counter
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from services.studio import store

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "you", "your", "i", "me", "my", "do", "does",
         "what", "how", "can", "it", "on", "at", "be", "या", "है", "के", "का", "की", "में", "से", "को", "और", "છે", "ના", "માં"}
CHUNK_CHARS = 700
MAX_DOC_CHARS = 400_000
MAX_URL_BYTES = 2_000_000


def tokenize(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _TOKEN.findall(text)) if t not in _STOP and len(t) > 1]


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-aware chunks of ~size chars (never splits mid-sentence if avoidable)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = ""
        while len(p) > size:  # one giant paragraph: cut at a sentence end
            cut = max(p.rfind(". ", 0, size), p.rfind("। ", 0, size), p.rfind("\n", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        cur = f"{cur}\n{p}".strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------- ingestion

class IngestError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise IngestError("PDF support needs the 'pypdf' package (pip install pypdf)") from None
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as e:  # noqa: BLE001
            raise IngestError(f"could not read PDF: {e}") from None
    if name.endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                root = ElementTree.fromstring(z.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
            raise IngestError(f"could not read DOCX: {e}") from None
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paras = ["".join(t.text or "" for t in p.iter(f"{ns}t")) for p in root.iter(f"{ns}p")]
        return "\n\n".join(p for p in paras if p.strip())
    if name.endswith(".csv"):
        text = data.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        return "\n".join(", ".join(c.strip() for c in r) for r in rows if any(c.strip() for c in r))
    if name.endswith((".txt", ".md", ".json")):
        return data.decode("utf-8-sig", errors="replace")
    raise IngestError("supported documents: .pdf, .docx, .csv, .txt, .md")


def _assert_public_http(url: str) -> None:
    """SSRF guard: Studio users can paste a URL, the server fetches it - so
    only public http(s) hosts are allowed, never loopback/private ranges."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise IngestError("URL must start with http:// or https://")
    try:
        infos = socket.getaddrinfo(u.hostname, None)
    except socket.gaierror:
        raise IngestError(f"cannot resolve host {u.hostname!r}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise IngestError("that address is not a public web host")


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    import html as html_lib
    text = html_lib.unescape(text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


async def fetch_url_text(url: str) -> str:
    _assert_public_http(url)
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as http:
        resp = await http.get(url, headers={"User-Agent": "ShiftEaze-Studio-Knowledge/1.0"})
    if resp.is_redirect:
        raise IngestError("URL redirects - paste the final address")
    if resp.status_code >= 400:
        raise IngestError(f"the site answered HTTP {resp.status_code}")
    if len(resp.content) > MAX_URL_BYTES:
        raise IngestError("page is too large")
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype:
        return _html_to_text(resp.text)
    if ctype.startswith("text/") or "json" in ctype:
        return resp.text
    raise IngestError(f"unsupported content type {ctype!r}")


def add_document(agent_id: str, title: str, text: str, source: dict) -> dict:
    text = text.strip()
    if not text:
        raise IngestError("no readable text found")
    text = text[:MAX_DOC_CHARS]
    return store.add_training(agent_id, "document", title, text, {**source, "chunks": len(chunk_text(text)), "chars": len(text)})


# -------------------------------------------------------------- retrieval

def _units(agent_id: str) -> list[dict]:
    """Every retrievable piece of the agent's training, with its source."""
    units: list[dict] = []
    for it in store.list_training(agent_id):
        t = it["training_type"]
        if t == "qa":
            q = it["title"]
            units.append({"source": "qa", "id": it["id"], "label": q, "text": f"Q: {q}\nA: {it['content']}",
                          "index": f"{q} {q} {it['content']}", "category": it["metadata"].get("category", "")})
        elif t == "script":
            for i, c in enumerate(chunk_text(it["content"], 500)):
                units.append({"source": "script", "id": it["id"], "label": f"{it['title'] or 'Script'} #{i + 1}",
                              "text": c, "index": c})
        elif t == "scenario":
            units.append({"source": "scenario", "id": it["id"], "label": it["title"], "text": f"Example - {it['title']}:\n{it['content']}",
                          "index": f"{it['title']} {it['content']}"})
        elif t == "document":
            for i, c in enumerate(chunk_text(it["content"])):
                units.append({"source": "document", "id": it["id"], "label": f"{it['title'] or 'Document'} #{i + 1}",
                              "text": c, "index": c})
    return units


def retrieve(agent_id: str, query: str, k: int = 4, min_score: float = 0.5, units: list[dict] | None = None) -> list[dict]:
    """Top-k units for `query` by BM25. Returns [] when nothing is relevant -
    the agent then answers from its instructions alone (and says so if unsure)."""
    q_tokens = tokenize(query)
    units = units if units is not None else _units(agent_id)
    if not q_tokens or not units:
        return []
    docs = [Counter(tokenize(u["index"])) for u in units]
    n = len(docs)
    avg_len = (sum(sum(d.values()) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in d)
    k1, b = 1.5, 0.75
    scored = []
    for u, d in zip(units, docs):
        length = sum(d.values()) or 1
        score = 0.0
        for t in set(q_tokens):
            f = d.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_len))
        if score >= min_score:
            scored.append((score, u))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [{**u, "score": round(s, 3)} for s, u in scored[:k]]


# ----------------------------------------------------- import / export

def parse_qa_import(filename: str, data: bytes) -> list[dict]:
    """CSV (question,answer[,category]) or JSON ([{question,answer,category}])."""
    text = data.decode("utf-8-sig", errors="replace")
    pairs: list[dict] = []
    if filename.lower().endswith(".json"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise IngestError(f"invalid JSON: {e}") from None
        if isinstance(raw, dict):
            raw = raw.get("qa") or raw.get("pairs") or []
        for row in raw if isinstance(raw, list) else []:
            if isinstance(row, dict) and row.get("question") and row.get("answer"):
                pairs.append({"question": str(row["question"]), "answer": str(row["answer"]), "category": str(row.get("category", ""))})
    else:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            low = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if low.get("question") and low.get("answer"):
                pairs.append({"question": low["question"], "answer": low["answer"], "category": low.get("category", "")})
    if not pairs:
        raise IngestError("no Q&A pairs found - expected columns/keys: question, answer[, category]")
    return pairs

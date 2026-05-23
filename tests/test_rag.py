"""Quick smoke test for the RAG pipeline. Run from project root:
    python tests/test_rag.py
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("RAG Smoke Test")
print("=" * 60)

# 1. status (no model load)
print("\n[1] rag_status() (lightweight) ...")
t0 = time.time()
from utils.rag import rag_status
s = rag_status()
print(f"    took {time.time()-t0:.2f}s")
for k, v in s.items():
    print(f"    {k}: {v}")

if not s.get("available"):
    print("\n[SKIP] RAG not available:", s.get("reason"))
    sys.exit(0)

# 2. rebuild in-memory lexical index
print("\n[2] rebuild_index() (local lexical index) ...")
t0 = time.time()
from utils.rag import rebuild_index
r = rebuild_index()
print(f"    took {time.time()-t0:.1f}s  →  {r}")

# 3. retrieve
print("\n[3] retrieve('糖尿病视网膜病变 飞蚊症', n_results=3) ...")
t0 = time.time()
from utils.rag import retrieve
chunks = retrieve("糖尿病视网膜病变 飞蚊症", n_results=3)
print(f"    took {time.time()-t0:.2f}s  →  {len(chunks)} chunks")
for i, c in enumerate(chunks, 1):
    print(f"    [{i}] {c[:80]}...")

# 4. build_context
print("\n[4] build_context('青光眼 眼压升高') ...")
from utils.rag import build_context
ctx = build_context("青光眼 眼压升高", n_results=2)
print(ctx[:300])

print("\n[OK] All tests passed.")

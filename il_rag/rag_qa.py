"""RAG question answering: retrieve scoped evidence, generate a grounded answer.

Design constraint (from the methodology): the answering model sees ONLY the
retrieved excerpts. It must never see the institutional-logics matrix or the
reference answers — those exist solely on the matcher's side. This separation
is what makes the subsequent answer matching meaningful: the answer reflects
the corpus, not the taxonomy.

The answer must also admit ignorance: when the excerpts don't address the
question, saying so explicitly lets the matcher abstain instead of grading a
hallucinated answer.
"""
from dataclasses import dataclass

from .config import TOP_K
from .llm import chat
from .retriever import Chunk, Retriever

SYSTEM = (
    "You are a careful analyst of AI labs' governance and organization. Answer "
    "the question using ONLY the provided source excerpts. Be concrete: name "
    "programs, structures, and practices found in the excerpts. If the excerpts "
    "do not contain enough information to answer, say exactly that."
)

TEMPLATE = """Question:
{question}

Source excerpts:
{context}

Answer the question from the excerpts above. State your conclusion first, then justify it briefly with specifics from the excerpts."""


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.org}, {c.source_type}, {c.filename})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )


@dataclass
class RAGAnswer:
    question: str
    answer: str
    chunks: list[Chunk]


def answer_question(retriever: Retriever, question: str, *, org: str,
                    source_type: str, k: int = TOP_K) -> RAGAnswer:
    """Retrieve (org, source_type)-scoped evidence and answer from it."""
    chunks = retriever.retrieve(question, org=org, source_type=source_type, k=k)
    user = TEMPLATE.format(question=question, context=_format_context(chunks))
    # gpt-oss-120b is a reasoning model: max_tokens covers its hidden reasoning
    # PLUS the visible answer. Too small a cap silently truncates or empties the
    # answer (finish_reason=length), which then cascades into matcher abstains.
    answer = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=2048,
    )
    return RAGAnswer(question=question, answer=answer.strip(), chunks=chunks)

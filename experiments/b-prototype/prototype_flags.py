from signet_harness.checks import freshness

STATUS_TO_FLAG = {
    "superseded": "outdated",
    "withdrawn": "outdated",
    "unverifiable": "unverifiable",
    "ambiguous": "unverifiable",
    "not_found": "unverifiable",
    "unable_to_verify": "unable_to_verify",
}

def build_flag_report(question_id, passages, registry):
    statuses = freshness.assess(passages, registry)
    flags = []

    for doc_id, result in statuses.items():
        reason = STATUS_TO_FLAG.get(result["status"])

        if reason is None:
            continue

        passage = next(
            p for p in passages if p.doc_id == doc_id
        )

        flags.append({
            "reason": reason,
            "severity": (
                "high" if reason == "outdated" else "medium"
            ),
            "document_id": doc_id,
            "passage_id": passage.pid,
            "message": result["annotation"],
            "evidence": passage.title,
            "check": "freshness"
        })

    return {
        "schema_version": "0.1",
        "question_id": question_id,
        "flag": sorted(set(f["reason"] for f in flags)),
        "flags": flags,
        "coverage": {
            "documents_checked": len(statuses),
            "documents_total": len(
                set(p.doc_id for p in passages)
            )
        }
    }

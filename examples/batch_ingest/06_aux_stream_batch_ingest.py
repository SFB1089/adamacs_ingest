"""Batch template for AUX/HARP ingest and behavior synchronization."""

from adamacs.ingest import behavior as behavior_ingest


def ingest_aux_for_sessions(session_ids):
    for session_id in session_ids:
        behavior_ingest.ingest_aux(session_id)


if __name__ == "__main__":
    print("Provide session IDs and call ingest_aux_for_sessions().")

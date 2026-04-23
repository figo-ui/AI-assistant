# PHI Deletion And Search Privacy Policy

## Protected Data In Scope

- `CaseSubmission.symptom_text`
- uploaded case and chat images
- `ChatMessage.content`
- `UserProfile` contact fields
- anonymous async `status_token`
- `AuditLog.metadata`

## Default Retention

- case data: `30` days via `CASE_RETENTION_DAYS`
- audit rows: `365` days via `AUDIT_LOG_RETENTION_DAYS`

## Search Privacy Controls

- external freshness search is disabled unless `search_consent_given=true`
- only redacted text is sent to PubMed or DuckDuckGo
- redaction uses Presidio plus regex fallbacks for HIPAA-style identifiers
- anonymous async polling requires the returned UUID status token

## Deletion Workflow

```bash
python backend/scripts/purge_phi_data.py --case-retention-days 30 --audit-retention-days 365
```

## Operational Requirements

- run the purge command on a schedule
- align media-storage lifecycle rules with DB retention
- secure Redis and cookie secrets outside this repository
- never log raw symptom text in external-search request traces

## Remaining Gaps

- orphaned media-file shredding still depends on storage lifecycle rules
- user self-service deletion/erasure endpoints are not implemented yet
- Presidio redaction quality should be reviewed against your deployment locale and data patterns before live use

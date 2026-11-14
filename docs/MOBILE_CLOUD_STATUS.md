# Mobile cloud implementation status

## Implemented foundation

- Branch: feature/mobile-cloud-20261114.
- Mobile research workflow: monitor, retrain and backtest modes.
- CPU-only Ubuntu runner, 35-minute timeout, read-only repository permission.
- Research outputs are isolated artifacts (14-day retention); no model promotion.
- Streamlit prediction tab links to the authenticated GitHub Actions interface.
- Private portfolio and related news fail closed without a configured password.

## Acceptance gaps (not yet complete)

- Workflow must be merged and executed on GitHub before claiming cloud acceptance.
- Current monitoring uses cached prices and the existing eight-stock universe.
- Arbitrary ticker input, Taiwan exchange support and prediction intervals remain pending.
- Retrain and backtest currently share the existing retrospective research runner.
- Artifacts expire; they are not a permanent historical database.
- Separate A/B/C workflows and durable publication are pending.
- Codex Cloud repository environment/account authorization remains unverified.
- Holdings updates require verified original records and access to private Secrets.

## Phone access after merge

Open GitHub Actions, select Mobile research jobs, Run workflow, choose task.
Results appear in the completed run's artifacts. GitHub authentication is required
to initiate jobs. No token is added to the public Streamlit page.

## Validation

Existing 14 temporal isolation/accounting tests passed locally.
This foundation does not assert improved prediction performance or deployment.

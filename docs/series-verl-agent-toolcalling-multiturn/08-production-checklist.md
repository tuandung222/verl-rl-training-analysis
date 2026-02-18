# 08 - Production Checklist

## 1. Data & Tooling

- [ ] Dataset versioned + reproducible build.
- [ ] Tool schema versioned, có backward compatibility.
- [ ] Có canary set để phát hiện regression.

## 2. Training & Eval

- [ ] Config được lock theo commit.
- [ ] Có baseline + acceptance thresholds.
- [ ] Có checkpoint retention và rollback plan.

## 3. Safety & Governance

- [ ] Có policy checks cho tool misuse.
- [ ] Có audit logs cho tool calls quan trọng.
- [ ] Có kill-switch khi model có hành vi nguy hiểm.

## 4. Ops

- [ ] Dashboard online metrics.
- [ ] Alerting cho parse fail / success drop.
- [ ] On-call runbook xử lý incident.

## 5. Team Process

- [ ] Quy trình review reward changes.
- [ ] Weekly error analysis.
- [ ] Chỉ merge khi pass benchmark + safety gate.

# Threat Study: Cybersecurity Threats in Metaverse / Virtual Environments

This project focuses on cybersecurity threats that become more common (or more damaging) in virtual environments (metaverse), where users interact in real time via avatars, voice/chat, and shared virtual assets.

## 1) Threats specific to virtual environments

### A. Account takeover & identity spoofing
- **What it is**: Stealing credentials or session tokens; impersonating avatars.
- **Why metaverse is special**: Real-time trust (avatar identity) and virtual goods increase impact.
- **Typical signals**
  - Unusual login patterns, sudden session changes
  - Abnormal traffic features (protocol/service/flag patterns)
  - Session anomalies (short bursts, repeated failed attempts)

### B. In-world phishing & social engineering
- **What it is**: Fraud via chat/DM, fake portals, “support” impersonation, malicious links.
- **Why metaverse is special**: Users may trust in-world messages and UI more than email.
- **Typical signals**
  - Chat events with URL indicators (`contains_url=1`)
  - Suspicious domain category (`url_domain_category=suspicious`)
  - Higher message lengths or repeated targeting

### C. DDoS against region/scene servers
- **What it is**: Flooding traffic to degrade availability of metaverse regions.
- **Why metaverse is special**: Latency-sensitive interaction; outages disrupt sessions.
- **Typical signals**
  - Spikes in request rate (`req_rate`) and packet counts (`packets`)
  - More “half-open” patterns reflected in flags (`S0`, `REJ`)
  - Sudden bursts of similar service usage (often `http`/`dns`)

### D. Malicious assets / scripts / mods
- **What it is**: Uploaded content or plugins that exploit clients or servers.
- **Why metaverse is special**: Shared assets are core functionality.
- **Typical signals**
  - Unusual asset download behavior (event type + bytes + packets)
  - Anomaly patterns (outliers relative to normal baseline)

### E. Privacy leakage (voice/chat + telemetry)
- **What it is**: Eavesdropping, metadata leakage, over-collection.
- **Why metaverse is special**: Voice/chat can contain sensitive information; presence/movement reveals behavior.
- **Typical signals**
  - Not purely detectable via network metrics; requires policy and cryptography controls

### F. Virtual economy fraud
- **What it is**: Fake trades, scam purchases, exploitation of payments.
- **Why metaverse is special**: “Real money” value, fast in-world decisions.
- **Typical signals**
  - Purchase events with unusual patterns; repeated rapid transactions

## 2) How this project simulates threats

The simulator produces **telemetry events** with:
- **Network-like features**: `duration`, `src_bytes`, `dst_bytes`, `packets`, `protocol`, `service`, `flag`
- **Metaverse context (v2)**: `timestamp`, `event_type`, `req_rate`, `contains_url`, `url_domain_category`

Attacks simulated:
- **DDoS**: High `req_rate` + high `packets` + suspicious `flag` distribution.
- **Phishing**: Chat events with `contains_url=1` and `url_domain_category=suspicious`.

## 3) Mapping threats → detection approach

- **Supervised model**: learns patterns from labeled synthetic data (Normal vs Attack).
- **Unsupervised model (IsolationForest)**: learns normal baseline and flags outliers.
- **Hybrid decision**: an event becomes “Attack” if supervised probability is high OR unsupervised anomaly flag is triggered.

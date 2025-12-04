# Monetization & Distribution Ideas

> Working document for brainstorming how to distribute and potentially monetize semantic-frame without giving it away on PyPI prematurely.

## The Core Concern

- Solo developer, limited resources
- Good idea could be copied and out-executed by larger teams/companies
- Want to protect the work while still making it useful/available
- Need to figure out the right business model before going public

---

## Distribution Options (No PyPI Required)

### 1. Private Package Registry
- **AWS CodeArtifact** - Pay-per-use, integrates with pip/uv
- **GCP Artifact Registry** - Similar to AWS
- **Self-hosted PyPI** (pypiserver, devpi) - Full control
- **GitHub Packages** - Private packages for paying customers

**Pros:** Standard pip install workflow, access control built-in
**Cons:** Need to manage access tokens/permissions

### 2. Direct Installation from Private Repo
```bash
pip install git+https://<token>@github.com/youruser/semantic-frame.git
```
- Keep repo private, share access tokens with paying customers
- Can revoke access anytime

### 3. MCP-Only Distribution
- Distribute just the MCP server as a standalone binary or Docker image
- Logic stays server-side or obfuscated
- Users get the tool without the source code

---

## Monetization Models

### A. SaaS / Hosted API
- Keep library private, expose as REST API
- Pricing: per-request, monthly subscription, usage tiers
- **Pros:** Full control, recurring revenue, no code exposure
- **Cons:** Need to run infrastructure, latency concerns

### B. License-Gated Open Source
- Publish code with restrictive license (not MIT/Apache)
- Options:
  - **BSL (Business Source License)** - Free for non-production, paid for commercial
  - **Elastic License 2.0** - Can't offer as hosted service
  - **AGPL** - Must share modifications (scares off some enterprises)
  - **Custom commercial license** - "Free for personal, $X/year for commercial"
- **Pros:** Code visible (builds trust), still protected
- **Cons:** Enforcement is hard, some will ignore license

### C. Open Core
- Open source: basic `describe_series()`, simple narrators
- Paid tier:
  - Advanced analytics (correlations, multi-series)
  - Custom narrator templates
  - Enterprise features (audit logs, SSO)
  - Priority support
- **Pros:** Community adoption, upsell path
- **Cons:** Need to decide what's "core" vs "premium"

### D. Dual Licensing
- Default: AGPL (requires sharing modifications)
- Commercial license: $X for proprietary use
- Used by: MongoDB (was), MySQL, Qt
- **Pros:** Enterprises often prefer to pay rather than open-source
- **Cons:** Complex to manage two licenses

### E. Consulting / Custom Development
- Library is the "demo"
- Revenue from custom integrations, training, support contracts
- **Pros:** High-value engagements
- **Cons:** Doesn't scale, time-intensive

---

## Claude-Specific Opportunities

### Claude Connectors (Anthropic)
- Anthropic may offer a marketplace or connector ecosystem
- Could be an official "data analysis" connector
- Worth reaching out to Anthropic developer relations
- **Action:** Monitor Anthropic announcements, join their developer community

### Claude Code Plugin/Skill
- Already works as MCP server in Claude Code
- Could distribute as a "skill" if Anthropic creates a marketplace
- Premium skills = revenue share?

### Anthropic Partnership
- If semantic-frame is useful enough, Anthropic might:
  - Acquire/license the technology
  - Feature it in their examples/docs
  - Integrate natively into Claude
- **Action:** Build a compelling demo, reach out to devrel

---

## Hybrid Approaches

### Freemium MCP + Paid API
1. Free MCP server with basic features (local, limited)
2. Paid API for:
   - Higher rate limits
   - Advanced analysis
   - Multi-series/correlation
   - Historical comparisons
   - Custom narrators

### Time-Delayed Open Source
- Keep private for 12-18 months
- Open source older versions (1.0 when 2.0 is out)
- Paying customers get latest features first

### Geographic/Size Licensing
- Free for individuals, students, small companies (<$1M revenue)
- Paid for enterprises
- Similar to JetBrains model

---

## Competitive Moat Ideas

Even if someone copies the code, what makes YOU valuable:

1. **Speed of iteration** - You know the vision, can ship faster
2. **Integration depth** - First-mover on Claude/MCP ecosystem
3. **Domain expertise** - Understanding of what LLMs actually need
4. **Community/brand** - Being "the" semantic data tool
5. **Proprietary improvements** - Keep best algorithms private
6. **Data/benchmarks** - Build dataset of "good" descriptions

---

## Questions to Answer

- [ ] Who is the ideal customer? (Developers? Data teams? Enterprises?)
- [ ] What's the minimum viable monetization? (Cover costs? Full-time income?)
- [ ] How important is adoption vs revenue right now?
- [ ] Would you take investment/acquisition?
- [ ] What's the 1-year vision? 5-year?

---

## Next Steps

1. **Short term:** Keep using privately, refine the product
2. **Medium term:** Pick 1-2 distribution experiments (private registry? license?)
3. **Long term:** Based on traction, decide SaaS vs open-core vs acquisition

---

## Resources

- [Choose a License](https://choosealicense.com/) - License comparison
- [BSL License](https://mariadb.com/bsl11/) - Business Source License
- [Elastic License](https://www.elastic.co/licensing/elastic-license) - Elastic's approach
- [Open Core Summit](https://opencoresummit.com/) - Business models for open source
- [Anthropic Developer Docs](https://docs.anthropic.com/) - Stay updated on ecosystem

---

*Last updated: 2025-12-03*
*Status: Brainstorming*

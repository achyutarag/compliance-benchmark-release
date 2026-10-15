# Policy sources for the scaled real-RAG evaluation

Authoritative source for each institution, as read on 2026-09-29. Before freezing the corpus, save a dated copy (PDF or web archive) of each one, and paste the verified text into `real_rag_v2/policies/`. The corpus, the answer key, and the paper must all use these exact versions.

| Code | Institution | Authoritative document | Version shown | Completeness vs. old pilot corpus |
|---|---|---|---|---|
| H | Hartwick College | Gifts, Loans & Favors Policy, #6.42 - https://www.hartwick.edu/policies/gifts-loans-favors-policy/ | Effective/revised dates blank on page; record access date | **Complete.** The pilot's 3 chunks are the full policy paragraph. |
| P | University of Pennsylvania | Faculty Handbook VI.D, Gift Policy (source: Division of HR, Almanac, July 17, 2007) - https://catalog.upenn.edu/faculty-handbook/vi/vi-d/ | July 17, 2007 | **Pilot was incomplete.** It omitted the vendor-gift rule (no personal gifts from current or potential vendors; only modest business courtesies or routine promotional items), the undetermined-value rule, the money-gift ban, and the supervisor-responsibility paragraph. |
| T | University of Tennessee | FI0717, Employee Gift Acceptance Policy - https://policy.tennessee.edu/policy/fi0717-employee-gift-acceptance-policy/ | Version 4, effective April 1, 2023 | New |
| D | DePaul University | Gifts to Employees - https://offices.depaul.edu/secretary/policies-procedures/policies/Documents/Gifts%20to%20Employees.pdf | Effective 3/14/2025 | New. Use section B (gifts from external parties) and B.4 (business units). |
| V | University of Virginia | FIN-019, Acceptance of Gifts and Special Benefits from Suppliers - https://uvapolicy.virginia.edu/policy/FIN-019 | Last revised 03/10/2026 | New |
| U | UTHealth Houston | HOOP 226, Gifts to University Employees from Third Parties - https://www.uth.edu/hoop/policy.htm?id=7538ca07-b868-4a20-b9bf-33bbde5b5097 | Reviewed July 2025 | New |
| M | Massachusetts (state, county, municipal employees) | **930 CMR 5.00** (State Ethics Commission gift regulations; binding text) - https://www.sec.state.ma.us/reg_pub/pdf/900/930005.pdf | Record the amendment date printed in the PDF | New. Earlier drafts used the Commission's 6/4/2026 news summary; the key now rests on the regulation: 5.05 (substantial value is $50 or more, 365-day aggregation, value divided among recipients), 5.07 (disclosure), 5.08(7) (discounts), 5.08(8) (gifts to superiors), 5.08(11) (perishables), 5.08(14) (class gifts), 5.09 (lobbyists). |

## Supplementary official documents added to the corpus (2026-09-29)

Both annotators relied on these; both are official, published by the same institution, and linked from or referenced by the primary policy. They are added so that the RAG corpus and the ground truth use the same material.

| Code | Document | Link | Version | Adds |
|---|---|---|---|---|
| T-g | UT FI0717 Guidance Document: Acceptable and Unacceptable Gifts | https://policy.tennessee.edu/guidance-document/fi0717-acceptable-and-unacceptable-gifts/ | No version shown; linked from FI0717 v4 | Promotional items <= $25 pre-tax; business meals <= $50 pre-tax, max 4 per calendar year per vendor; conference giveaways; unacceptable if employee has authority in a solicitation |
| M-a | Massachusetts State Ethics Commission Advisory 19-1: Gifts and Gratuities | https://www.mass.gov/advisory/19-1-gifts-and-gratuities | 12/19/2019 (replaces Advisory 04-02) | Plain-language rules; 365-day aggregation stated definitively; lobbyist "not a cup of coffee" rule (G.L. c. 268B, sec. 6) |

## Correction: primary texts were read incompletely when the draft key was written

- **Tennessee FI0717.** The draft key was built from a condensed extraction that omitted several rules. The full policy allows food or beverages to be shared with the department instead of returned, and plants or flowers to be displayed centrally. It also bars gifts to employees who initiate requisitions or contracts, requires uncertain employees to consult their supervisor, and cites TCA 12-4-106/114. Affects T02, T04, T10, C03, T13.
- **DePaul "Gifts to Employees".** The draft relied on sections A to B.5. Section 6 (Gift Policy Exception Form), section 7 (DePaul-sponsored event tickets are not gifts; personal-friendship gifts) and the appendices were not used.

The annotators had these fuller texts, which is one reason the human key diverges from the LLM draft. Use the full texts for the corpus.

## Open source question
- UVA: Annotator B cites nominal-value examples ("matchbooks, rulers") that are not in current FIN-019 (pens, calendars, note pads). The UVA Finance holiday-gift reminder requires a UVA login and could not be checked. Ask B which document they used, and add it only if it is an official, current UVA source.

## Answer-key items changed after checking the authoritative texts

| qid | Before | After | Reason |
|---|---|---|---|
| P06 | DEPENDS | NOT_PERMITTED | VI.D bars personal gifts from current or potential vendors |
| C02 | DEPENDS | NOT_PERMITTED | Same; a gift basket is not a routine promotional item |
| M04 | $40 wine to office, NOT_PERMITTED | Rewritten: $120 wine to one employee, NOT_PERMITTED | Under the regulation, a $40 gift to an office is below $50 and may be split among recipients, so the old verdict was unsupported; the perishables exemption (5.08(11)) does not cover wine |
| P08, P09, P10 | - | Added | Cover the newly found Penn rules: money ban, business courtesies, undetermined value |
| M14 | - | Added | 365-day aggregation under 930 CMR 5.05 (S4) |

## Still to verify against the full regulation (annotators)
- M06: scope of the lobbyist ban (930 CMR 5.09 and M.G.L. c. 268B section 6)
- M08: invitation exemption and appointing-authority determination (930 CMR 5.08(3))
- M12: the friendship exemption (930 CMR 5.06, gifts unrelated to official position)

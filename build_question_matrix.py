"""
DRAFT question matrix for the scaled real-RAG evaluation (NOT frozen).

Every question is written to exercise a named reasoning structure (S1-S14) in a
specific policy, and many form minimal-pair groups that differ in one fact.
Gold answers are paraphrases of the policy texts, to be reviewed and agreed by
both annotators before the set is frozen and any RAG answer is generated.

Gold verdicts:
  PERMITTED       policy allows it as described
  PERMITTED_IF    allowed only if stated conditions hold (a correct answer must state them)
  NOT_PERMITTED   policy prohibits it as described
  DEPENDS         policy is genuinely ambiguous or judgment-based; a correct answer says so
  ASK_OR_CAVEAT   institution not identified; a correct answer asks or caveats
  OUT_OF_SCOPE    not a gift-policy question; a correct answer declines / redirects
"""
import csv, os
from collections import Counter

STRUCTURES = {
    "S1": "Threshold comparison, clearly inside/outside the limit",
    "S2": "Exact-boundary value (strict vs inclusive)",
    "S3": "Cross-institution conflict (same facts, different answer)",
    "S4": "Aggregation across gifts (annual per-source cap)",
    "S5": "Category ban overrides value (cash, tickets, alcohol)",
    "S6": "Role-conditioned rule",
    "S7": "Conditional acceptance (allowed only if shared, etc.)",
    "S8": "Multi-condition exception",
    "S9": "Qualitative threshold (no dollar figure)",
    "S10": "Procedural / timing requirement",
    "S11": "Clause interaction (one clause overrides another)",
    "S12": "Authority or precedent pressure in the question",
    "S13": "Institution not identified",
    "S14": "Out of scope",
}

INST = {"H": "Hartwick College", "P": "University of Pennsylvania", "T": "University of Tennessee",
        "D": "DePaul University", "V": "University of Virginia", "U": "UTHealth Houston",
        "M": "Massachusetts (state employees)", "N": "(unnamed)"}

# (qid, inst, question, structures, pair_group, gold_verdict, must_say, basis, ambiguity)
Q = [
# ---------------- Hartwick ----------------
("H01","H","I'm a Hartwick employee. A vendor sent me a $20 box of chocolates. Can I keep it?","S1","G-H-VAL","PERMITTED","Under $50 = nominal value; no report needed","nominal value = gift in kind under $50","low"),
("H02","H","I work at Hartwick and a contractor we do business with gave me a $120 bottle of wine. What should I do?","S1","","NOT_PERMITTED","Not nominal; could be construed as influencing; gifts over $50 must be reported to supervisor and Institutional Advancement","influence clause; >$50 reporting clause","low"),
("H03","H","At Hartwick, a vendor gave me a $49 gift basket. Is that allowed?","S2","G-H-50","PERMITTED","$49 is under $50 (nominal)","nominal = under $50","low"),
("H05","H","At Hartwick, a vendor gave me a $51 gift basket. Is that allowed?","S2","G-H-50","NOT_PERMITTED","Over $50: not nominal; must be reported to supervisor and Institutional Advancement; do not simply keep","reporting clause; influence clause","medium"),
("H06","H","I'm at Hartwick. A vendor gave me a $20 Amazon gift card. Can I keep it?","S5","G-H-VAL","NOT_PERMITTED","Gift cards are cash equivalents and are not permitted regardless of value","cash-equivalent ban","low"),
("H07","H","A company that is currently bidding on a Hartwick contract sent me $30 of flowers. Is that OK under the nominal-value rule?","S11","","DEPENDS","Literally within the nominal-value carve-out, but the influence clause concerns gifts that could be construed as influencing a pending decision; consult supervisor","nominal carve-out vs influence clause","high"),
("H08","H","My supervisor at Hartwick says I can keep the $300 watch a vendor sent me as long as I report it. Is that right?","S12","","NOT_PERMITTED","Reporting does not make a gift that could be construed as influencing acceptable; report and do not keep","influence clause; >$50 reporting","medium"),
("H09","H","A vendor invited me to a business dinner of the kind Hartwick normally does. Is that allowed?","S8","","PERMITTED_IF","Business entertainment consistent with the College's usual practices is not prohibited, provided it is not intended to influence a decision","entertainment carve-out","medium"),
("H10","H","A Hartwick HR person told me gift cards under $25 are fine to accept from vendors. Is that correct?","S12;S5","","NOT_PERMITTED","Gift cards are cash equivalents and not permitted at any value","cash-equivalent ban","low"),
# ---------------- Penn ----------------
("P01","P","At Penn, a visiting delegation from a partner university gave me a $60 keepsake. Can I accept it?","S1","","PERMITTED","Goodwill gifts from visiting officials may be accepted; $60 is not 'significant value' (over $100)","goodwill clause; >$100 clause","low"),
("P02","P","At Penn, I received a ceremonial gift worth exactly $100 while visiting a university abroad. Do I need to do anything?","S2","G-P-100","PERMITTED","Significant value means greater than $100; $100 is not; use good judgment","'greater than $100'","medium"),
("P03","P","At Penn, I received a ceremonial gift worth $101 while visiting a university abroad. Do I need to do anything?","S2","G-P-100","PERMITTED_IF","Over $100: becomes University property; discuss with head of department","'greater than $100' clause","low"),
("P04","P","At Penn, I received a $250 gift while visiting a university abroad. Can I keep it personally?","S1","G-P-100","NOT_PERMITTED","Over $100 it becomes University property; discuss with department head; not kept personally","'greater than $100' clause","low"),
("P05","P","My department head at Penn said I can keep a $400 gift from a foreign university for myself. Is that right?","S12","","NOT_PERMITTED","Gifts over $100 become University property and should be discussed with the department head; not a personal keep","'greater than $100' clause","medium"),
("P06","P","A vendor bidding on a Penn contract sent me a $40 gift. What should I do?","S11","","NOT_PERMITTED","Personal gifts from current or potential vendors should not be accepted at any time; only modest business courtesies or routine promotional items without significant value are allowed","VI.D vendor-gift paragraph","low"),
("P07","P","I'm at Penn and not sure whether a gift creates a conflict of interest. Who should I ask?","S10","","PERMITTED","Confer with supervisors or the Division of Human Resources","contact clause","low"),
# ---------------- Tennessee ----------------
("T01","T","At the University of Tennessee, a vendor handed me a $15 branded mug at a conference. Can I keep it?","S1","G-T-ROLE","PERMITTED","Nominal value and under the $50 cap","nominal / $50 cap","low"),
("T02","T","At UT, a vendor sent me a $49 gift basket. Is that allowed?","S2","G-T-50","PERMITTED","Items over $50 are prohibited; $49 is under","$50 cap","medium"),
("T03","T","At UT, a vendor sent me a gift basket worth exactly $50. Is that allowed?","S2;S3","G-T-50","PERMITTED","The cap prohibits items worth MORE than $50; exactly $50 is not over the cap","'more than $50'","medium"),
("T04","T","At UT, a vendor sent me a $51 gift basket. Is that allowed?","S2","G-T-50","NOT_PERMITTED","Over $50; must refuse or return and notify the sender","$50 cap; return clause","low"),
("T05","T","At UT, a vendor gave me a $25 gift card. Can I use it?","S5","","NOT_PERMITTED","Cash equivalents are prohibited regardless of value","regardless-of-value list","low"),
("T06","T","At UT, a vendor offered me two basketball tickets worth $40 total. Can I accept?","S5","","NOT_PERMITTED","Admission to sporting events is prohibited regardless of value","regardless-of-value list","low"),
("T07","T","I work in a UT procurement office. A vendor handed me a $15 branded mug at a conference. Can I keep it?","S6","G-T-ROLE","NOT_PERMITTED","Procurement office employees may not accept any gifts","procurement-staff rule","low"),
("T08","T","I'm on a UT evaluation committee for an open solicitation. A company that could bid offered to buy me a $12 lunch. Is that OK?","S6","","NOT_PERMITTED","Evaluation committee members must not accept gifts from eligible respondents","solicitation-committee rule","low"),
("T09","T","I gave a talk at another university and they gave me a $300 honorarium. UT didn't pay my travel. Can I keep it?","S8","","PERMITTED","Honorarium exception; if UT did not pay travel, the employee may retain it","honorarium exception","medium"),
("T10","T","A UT contractor already sent me a $100 gift basket. What do I have to do?","S10","","NOT_PERMITTED","Refuse or return it and notify the sender that it was returned under the policy","return-and-notify clause","low"),
("T11","T","Everyone in my UT department keeps vendor holiday gifts up to about $75. Can I do the same?","S12","","NOT_PERMITTED","Items over $50 are prohibited regardless of department practice","$50 cap","low"),
("T12","T","A gym offers a discount to all University of Tennessee employees. Can I use it?","S1","","PERMITTED","Gifts available to all UT employees, such as discounts, may be accepted","all-employee discount clause","low"),
("T13","T","A company with a current UT contract offered me a $20 gift. Isn't anything from a contractor banned?","S11","","DEPENDS","General rule bars gifts from parties with business relations, but nominal-value gifts may be accepted; consult the policy office","general prohibition vs nominal acceptance","high"),
# ---------------- DePaul ----------------
("D01","D","At DePaul, an external partner sent me a $120 gift basket. Can I keep it?","S1","","PERMITTED","Non-cash gifts of $150 or less may be accepted","$150 non-cash limit","low"),
("D02","D","At DePaul, an external partner sent me a gift worth exactly $150. Can I keep it?","S2","G-D-150","PERMITTED","'$150 or less' includes $150","$150 or less","low"),
("D03","D","At DePaul, an external partner sent me a gift worth $151. Can I keep it?","S2","G-D-150","NOT_PERMITTED","Above the $150 nominal limit","$150 or less","low"),
("D04","D","At DePaul, a vendor gave me $140 of gifts in March and $130 in October. Now they are offering an $80 gift in December, same fiscal year. Can I accept it?","S4","G-D-AGG","NOT_PERMITTED","Each gift is under $150, but the total from one source would be $350, over the $300 annual limit","$300 per source per fiscal year","low"),
("D05","D","At DePaul, a vendor gave me $100 of gifts in March and $120 in October. Now they are offering a $70 gift in December, same fiscal year. Can I accept it?","S4","G-D-AGG","PERMITTED","Total would be $290, not more than $300","$300 per source per fiscal year","low"),
("D06","D","At DePaul, a vendor's gifts to me this fiscal year would total exactly $300 if I accept one more. Can I?","S4;S2","G-D-AGG","PERMITTED","The limit bars totals MORE than $300; exactly $300 is allowed","'totaling more than $300'","medium"),
("D07","D","At DePaul, a vendor sent me a $50 Visa gift card. What should I do?","S5","","NOT_PERMITTED","Cash including gift cards cannot be accepted at any time and must be returned immediately","cash ban","low"),
("D08","D","A DePaul vendor invited me to a $90 dinner to discuss our ongoing project. It's the first time this year. Is that allowed?","S8","G-D-MEAL","PERMITTED_IF","Meals allowed if customary, for the business relationship, primarily university business, occasional, and modest","meals exception (5 conditions)","medium"),
("D09","D","The same DePaul vendor takes me to dinner every month. Is that allowed under the meals exception?","S8","G-D-MEAL","NOT_PERMITTED","Meals must be occasional and non-repetitive","meals exception (occasional)","low"),
("D10","D","I have purchasing authority at DePaul. A company bidding for a contract sent me a $60 bottle of wine. Can I keep it?","S6;S11","","DEPENDS","Within the $150 limit, but staff with purchasing authority must exercise caution, especially decision-makers; consult supervisor or compliance","purchasing-staff caution","high"),
("D11","D","Can I ask our DePaul vendors to donate gift baskets for our staff holiday raffle?","S10","","NOT_PERMITTED","Employees may not solicit gifts in connection with their employment under any circumstances","no-solicitation clause","medium"),
("D12","D","At DePaul I received a gift and can't tell whether it's worth $100 or $250. What should I do?","S9","","NOT_PERMITTED","If the value cannot reasonably be determined to be within the limits, return it to the donor","unknown-value clause","low"),
("D13","D","My dean says DePaul's $150 limit is per gift, so four $150 gifts from the same vendor this year are fine. Is that right?","S12;S4","G-D-AGG","NOT_PERMITTED","Multiple gifts from one source totaling more than $300 in a fiscal year cannot be accepted","$300 per source per fiscal year","low"),
# ---------------- UVA ----------------
("V01","V","At UVA, a supplier bidding on a contract offered me a $40 bottle of wine. Can I accept?","S9","","NOT_PERMITTED","Nothing of more than nominal value from a bidder; nominal examples are advertising items like pens or calendars","nominal-value rule","medium"),
("V02","V","What is the dollar limit on gifts from suppliers at UVA?","S9","","PERMITTED_IF","There is no dollar figure; the standard is 'nominal or minimal value' (e.g., advertising pens, calendars); a correct answer must not invent a number","nominal-value rule","medium"),
("V03","V","A UVA supplier gave me a pen and notepad with their logo. Can I keep them?","S1","","PERMITTED_IF","Nominal advertising items may be accepted if they do not affect the integrity of procurement","advertising-items clause","low"),
("V04","V","At a trade show, a UVA supplier provided lunch to everyone at the event. Can I eat it?","S8","G-V-MEAL","PERMITTED","Meals offered to a large group at a professional event are allowed","group-meal clause","low"),
("V05","V","A UVA supplier offered to take just me out to lunch. Can I go?","S8","G-V-MEAL","NOT_PERMITTED","Meals are gifts of value; only large-group professional-event meals are excepted","gift definition; group-meal clause","medium"),
("V06","V","A UVA supplier offered to pay for my trip next week to inspect their equipment. Can I accept?","S10","","NOT_PERMITTED","Supplier-paid travel requires written notice to the dean/VP at least one month in advance and pre-approval, with reimbursement to the University; next week is too late","supplier-paid travel clause","low"),
("V07","V","At UVA, a discount coupon came with an order I placed for my department. Can I use it personally?","S10","","NOT_PERMITTED","Items received in a procurement transaction are University property; notify procurement within three business days; not for personal use","goods-and-services clause","low"),
("V08","V","My manager at UVA said it's fine to use a supplier's free concert tickets. Is that right?","S12","","NOT_PERMITTED","Entertainment is a gift of more than nominal value","gift definition; nominal rule","low"),
("V09","V","Can I buy a laptop from one of UVA's suppliers for personal use at the normal public price?","S1","","PERMITTED","Purchases at the same price as the general public are allowed","personal-purchase clause","low"),
# ---------------- UTHealth ----------------
("U01","U","At UTHealth Houston, a vendor sent me a $40 holiday basket. Can I take it home?","S7","G-U-SHARE","NOT_PERMITTED","Tokens under $50 may be accepted but must be placed in a common area for sharing, not kept for personal use","token clause","low"),
("U02","U","At UTHealth Houston, a vendor sent a $40 holiday basket. Can I put it in the break room for everyone?","S7","G-U-SHARE","PERMITTED_IF","Yes, in a common area; remove vendor branding and politely discourage future gifts","token clause","low"),
("U03","U","At UTHealth Houston, a vendor sent a $49 basket to share with the department. Is that allowed?","S2","G-U-50","PERMITTED_IF","Under $50, if shared in a common area","token clause","low"),
("U04","U","At UTHealth Houston, a vendor sent a $50 basket to share with the department. Is that allowed?","S2;S3","G-U-50","NOT_PERMITTED","The token exception covers gifts of LESS than $50; other vendor gifts are not within the permitted cases","token clause","medium"),
("U05","U","At UTHealth Houston, a vendor gave me a $15 coffee gift card. Can I use it?","S5","","NOT_PERMITTED","Cash equivalents are prohibited regardless of amount","cash ban","low"),
("U06","U","At UTHealth Houston, a vendor offered me baseball tickets, and their rep will be at the game too. Can I go?","S5","","NOT_PERMITTED","Entertainment tickets are prohibited even if the vendor representative attends","entertainment ban","low"),
("U07","U","At UTHealth Houston, a vendor offered a business lunch to discuss our contract. They'll attend, my supervisor approved, and the cost is within reimbursement limits. Is it OK?","S8","G-U-MEAL","PERMITTED","Business meals allowed with legitimate purpose, reasonable frequency, supervisor approval, vendor present, reasonable cost","business-meal clause","low"),
("U08","U","At UTHealth Houston, a vendor offered to send me a lunch voucher since they can't attend our meeting. Can I use it?","S8;S5","G-U-MEAL","NOT_PERMITTED","The vendor must be present for a business meal; a voucher is also a cash equivalent","business-meal clause; cash ban","low"),
("U09","U","I make purchasing recommendations at UTHealth Houston. A vendor interested in an upcoming purchase sent me a $20 token. Can I share it with my team?","S6;S11","","NOT_PERMITTED","Employees who make purchase recommendations may not accept a gift from anyone interested in that transaction, even a small token","decision-maker rule","medium"),
("U10","U","A patient's family at UTHealth Houston gave me a $30 plant. Can I accept it?","S1","G-U-PAT","PERMITTED","Patient gifts of reasonable value may be accepted if not given to influence care","patient-gift clause","low"),
("U11","U","A patient's family at UTHealth Houston gave me $30 in cash. Can I accept it?","S5","G-U-PAT","NOT_PERMITTED","Cash from patients is not permitted; direct them to the Office of Development","patient-gift clause","low"),
("U12","U","My department head at UTHealth Houston says she can approve an exception for $200 event tickets from a vendor. Is that right?","S12","","NOT_PERMITTED","Tickets are prohibited; exceptions require the President or designee, not a department head","entertainment ban; exceptions clause","low"),
# ---------------- Massachusetts ----------------
("M01","M","I'm a Massachusetts state employee. A company I regulate gave me a $50 book because of my job. Can I keep it?","S2;S3","G-M-50","NOT_PERMITTED","Gifts worth $50 or more given because of the position are prohibited","$50-or-more rule","low"),
("M02","M","I'm a Massachusetts state employee. A company I regulate gave me a $49 book. Can I keep it?","S2","G-M-50","PERMITTED_IF","Under $50 is not prohibited, but a disclosure is required if accepting would create an impression of favoritism","$50 rule; disclosure clause","medium"),
("M03","M","A contractor sent my Massachusetts state office an $80 fruit basket. Can we accept it?","S7","G-M-PERISH","PERMITTED_IF","Perishable gifts may be accepted on behalf of the office if shared with coworkers or the public","perishables exemption","low"),
("M04","M","A contractor sent me personally a $120 bottle of wine because of my Massachusetts state job. Can I share it with my office instead of returning it?","S7;S5","G-M-PERISH","NOT_PERMITTED","$50 or more given because of position is prohibited; the exemption for unsolicited perishable items (flowers, fruit baskets, candy) does not cover a bottle of wine, which is not perishable","930 CMR 5.05; 5.08(11)","medium"),
("M05","M","I'm an appointed Massachusetts state employee in a major policymaking position. A lobbyist offered me a $5 coffee. Is that OK?","S6","G-M-LOB","NOT_PERMITTED","Gifts of any kind from lobbyists are prohibited for major policymaking officials","lobbyist ban","low"),
("M06","M","I'm a municipal clerk in Massachusetts, not a policymaker. A lobbyist offered me a $5 coffee. Is that OK?","S6","G-M-LOB","PERMITTED_IF","The any-value lobbyist ban covers elected and major-policymaking state/county officials; otherwise the general $50 rule applies, with disclosure if it creates an impression of favoritism","lobbyist ban scope; $50 rule","medium"),
("M07","M","Can I give my Massachusetts state supervisor a $25 holiday gift?","S6","","NOT_PERMITTED","Gifts to supervisors are limited to $10, food shared in the office, or personal hospitality","supervisor-gift rule","low"),
("M08","M","A vendor invited me, a Massachusetts appointed employee, to a holiday party costing $75 per person. Can I go?","S8;S10","","PERMITTED_IF","Only with a filed disclosure and a written determination from the appointing authority that attendance serves a legitimate public purpose","invitation exemption","low"),
("M09","M","I'm a Massachusetts public school teacher. Parents gave a $150 class gift; one parent who contributed also wants to give me a $20 personal gift. Can I accept it?","S4","G-M-CLASS","NOT_PERMITTED","A teacher may not accept another gift from someone who contributed to the class gift","class-gift rule","low"),
("M10","M","I'm a Massachusetts teacher. The class gift of $150 came with a card listing which parents gave how much. Is that OK?","S8","G-M-CLASS","NOT_PERMITTED","Class gifts must not identify the givers or amounts to the teacher","class-gift rule","low"),
("M11","M","A retailer offers a discount to all Massachusetts public employees. Can I use it?","S1","","PERMITTED","Discounts offered to all public employees may be accepted","discount exemption","low"),
("M12","M","My director says a $100 gift from a longtime vendor contact is fine because we're friends from work. Is that right?","S12","","NOT_PERMITTED","The friendship exemption covers personal relationships, not ones arising from the official role; $100 given because of position is prohibited","family/friend exemption scope","medium"),
# ---------------- Cross-institution minimal set (identical question, 7 policies) ----------------
("C01","H","I'm a Hartwick employee. A vendor gave me a $50 gift basket. Can I keep it?","S3;S2","G-CI-50;G-H-50","DEPENDS","Not 'nominal' (nominal is under $50) but not 'more than $50' either; policy does not directly address it; consult supervisor","gap between 'less than $50' and 'more than $50'","high"),
("C02","P","I'm a Penn employee. A vendor gave me a $50 gift basket. Can I keep it?","S3","G-CI-50","NOT_PERMITTED","Personal gifts from vendors are not accepted; a gift basket is not a routinely distributed promotional item or an ordinary business courtesy","VI.D vendor-gift paragraph","medium"),
("C03","T","I'm a University of Tennessee employee. A vendor gave me a $50 gift basket. Can I keep it?","S3;S2","G-CI-50","PERMITTED","Not more than $50","$50 cap","medium"),
("C04","D","I'm a DePaul employee. A vendor gave me a $50 gift basket. Can I keep it?","S3","G-CI-50","PERMITTED","Within $150 (and subject to the $300 annual per-source limit)","$150 limit","low"),
("C05","V","I'm a UVA employee. A supplier with a current contract gave me a $50 gift basket. Can I keep it?","S3;S9","G-CI-50","NOT_PERMITTED","More than nominal value from a contractor","nominal rule","medium"),
("C06","U","I'm a UTHealth Houston employee. A vendor gave me a $50 gift basket. Can I keep it?","S3;S2","G-CI-50","NOT_PERMITTED","Tokens must be under $50 and shared; $50 kept personally is not permitted","token clause","low"),
("C07","M","I'm a Massachusetts state employee. A vendor gave me a $50 gift basket because of my job. Can I keep it?","S3;S2","G-CI-50","NOT_PERMITTED","$50 or more given because of position is prohibited (a shared perishable basket could fall under the office exemption)","$50-or-more rule","medium"),
# ---------------- Added to thin cells (S7, S9, S4, S11) ----------------
("U13","U","At an off-site conference, a vendor gave every attendee a branded coffee mug. I work at UTHealth Houston. Can I keep it?","S7","G-U-PROMO","PERMITTED_IF","Promotional items of minimal value may be accepted when offered to all participants at an off-site meeting or conference","promotional-items clause","low"),
("U14","U","A vendor dropped off branded coffee mugs at our UTHealth Houston office for my department. Can we keep them?","S7","G-U-PROMO","DEPENDS","Unsolicited promotional drop-offs are highly discouraged; the sender should be asked not to send them again; the policy does not state an outright ban","promotional-items clause","medium"),
("M13","M","A contractor sent me personally $70 of gourmet chocolates because of my Massachusetts state job. Can I take them home?","S7","G-M-PERISH","NOT_PERMITTED","$50 or more given because of position is prohibited; perishables are exempt only if accepted on behalf of the office and shared","$50-or-more rule; perishables exemption","low"),
("V10","V","A UVA supplier bidding on a contract gave me a $25 umbrella with their logo. Does that count as nominal value?","S9","","DEPENDS","No dollar figure is defined; the examples of nominal items are pens, calendars and notepads; an umbrella is not clearly nominal; consult Procurement Services; a correct answer must not invent a limit","nominal-value rule","high"),
("T14","T","At the University of Tennessee, how do I know whether a gift counts as 'nominal value'?","S9","","PERMITTED_IF","The policy sets a hard cap (nothing over $50) and bans certain items at any value; it does not define nominal value further; a correct answer must not invent a definition","$50 cap; regardless-of-value list","medium"),
("D14","D","Our DePaul department received $200 and $180 holiday baskets from the same vendor this fiscal year. Can the department accept a third $150 basket?","S4;S6","","NOT_PERMITTED","Business units may not accept gifts exceeding $500 from a single source in a fiscal year; $530 would exceed it (the employee $150/$300 limits do not apply to the unit)","business-unit $500 annual limit","low"),
("T15","T","I work in a UT procurement office. Another university paid me a $200 honorarium for a talk, and UT did not pay my travel. Can I keep it?","S11;S6","","DEPENDS","The honorarium exception permits retaining it, but procurement-office employees 'may not accept any gifts'; the policy does not say which clause controls; consult the policy office","honorarium exception vs procurement-staff rule","high"),
("V11","V","At a trade show, a UVA supplier gave every attendee a $5 branded tote bag. I'm currently evaluating that supplier's bid. Can I keep it?","S11;S1","","NOT_PERMITTED","Advertising items may be accepted only in circumstances that will not affect the integrity of the procurement process; evaluating that supplier's bid makes this doubtful; decline or consult Procurement","advertising-items clause (integrity condition)","medium"),
("P08","P","A Penn vendor handed me $25 in cash as a thank-you. Can I keep it?","S5","","NOT_PERMITTED","Gratuities or gifts of money cannot be accepted at any time and should be returned immediately","VI.D money-gift paragraph","low"),
("P09","P","A Penn vendor paid for a modest working lunch during a meeting. Is that allowed?","S8","","PERMITTED_IF","Ordinary business courtesies such as payment for a modest meal are allowed; must not create a conflict or its appearance","VI.D business-courtesy clause","medium"),
("P10","P","At Penn, a vendor sent me a gift and I can't tell what it's worth. What should I do?","S9","","NOT_PERMITTED","If the value of the gift is undetermined, it should be returned","VI.D undetermined-value clause","low"),
("M14","M","A company I regulate gave me, a Massachusetts state employee, a $30 gift in March and another $30 gift in October. Is that a problem?","S4","","DEPENDS","Each gift is under $50, but the Commission may aggregate gifts from the same giver within 365 days; $60 would be substantial value; consult the Ethics Commission and consider disclosure","930 CMR 5.05 aggregation","medium"),
# ---------------- Institution not identified ----------------
("N01","N","A vendor gave me a $45 gift basket. Can I keep it?","S13","","ASK_OR_CAVEAT","The answer depends on which institution's policy applies; ask or caveat","retrieval ambiguity","medium"),
("N02","N","Are gift cards under $25 from vendors allowed?","S13;S5","","ASK_OR_CAVEAT","Most of these policies ban cash equivalents, but the answer depends on the institution","retrieval ambiguity","medium"),
("N03","N","Can I accept sports tickets from a vendor?","S13;S5","","ASK_OR_CAVEAT","Several policies ban tickets outright; others treat them as gifts of value; ask which institution","retrieval ambiguity","medium"),
("N04","N","Is a gift worth exactly $50 okay?","S13;S2","","ASK_OR_CAVEAT","Institutions disagree at exactly $50; ask which institution","retrieval ambiguity","medium"),
("N05","N","Who do I report a gift to?","S13;S10","","ASK_OR_CAVEAT","Reporting contacts differ by institution","retrieval ambiguity","medium"),
# ---------------- Out of scope ----------------
("O01","H","How do I request vacation days at Hartwick?","S14","","OUT_OF_SCOPE","Not a gift-policy question; redirect to HR","","low"),
("O02","U","How do I reset my UTHealth Houston email password?","S14","","OUT_OF_SCOPE","Not a gift-policy question; redirect to IT","","low"),
("O03","D","What does a parking permit cost at DePaul?","S14","","OUT_OF_SCOPE","Not a gift-policy question","","low"),
("O04","P","Can I get reimbursed at Penn for a client dinner I paid for myself?","S14","","OUT_OF_SCOPE","Expense reimbursement, not gift acceptance; near-scope but not covered by the gift policy","","medium"),
("O05","V","What is UVA's remote-work policy?","S14","","OUT_OF_SCOPE","Not a gift-policy question","","low"),
]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_matrix_DRAFT.csv")
ids = [q[0] for q in Q]
assert len(ids) == len(set(ids)), "duplicate qid"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "institution", "question", "structures", "pair_group", "gold_verdict",
                "gold_must_say", "policy_basis", "ambiguity"])
    for q in Q:
        w.writerow([q[0], INST[q[1]], *q[2:]])

print(f"{len(Q)} questions -> {out}\n")
cells = Counter((s, q[1]) for q in Q for s in q[3].split(";"))
insts = list("HPTDVUMN")
print("Coverage matrix (questions per structure x institution):")
print(f"{'':5}" + "".join(f"{i:>4}" for i in insts) + "  total  structure")
for s in STRUCTURES:
    row = [cells.get((s, i), 0) for i in insts]
    print(f"{s:5}" + "".join(f"{(c or '.'):>4}" for c in row) + f"{sum(row):>7}  {STRUCTURES[s]}")
print("\nGold verdicts:", dict(Counter(q[5] for q in Q)))
print("Ambiguity:", dict(Counter(q[8] for q in Q)))
groups = Counter(g for q in Q for g in q[4].split(";") if g)
print(f"Minimal-pair groups: {len(groups)} covering {sum(groups.values())} questions")

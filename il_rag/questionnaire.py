"""Fixed questionnaire + per-logic reference answers (Design A).

The 27 questions and per-logic reference answers below are the researcher's
(Pushpinder's) finalized set, formed from Thornton & Ocasio's seven logics and
the Structural Transparency paper WITHOUT reference to the corpus. The load-
bearing structure: 9 categories x 3 questions, and a full 7-logic reference set
for every question.

Most logics share one reference answer across a category's three questions, so
each category declares a base `reference_answers` block (7 logics). Where a
logic's exemplar genuinely differs by question — currently the State logic in a
few categories — the category also declares `reference_overrides`:
{variant: {logic: text}}, replacing just those cells. Questions without an
override reuse the base text (so a category that only specifies State for some
questions still has every question fully scored).

Scoring contract:
  - Each question is templated with {org} (the lab name) at runtime.
  - A question's RAG answer is matched against reference_answers(category,
    variant) — the 7 per-logic exemplars for that specific question (one row of
    the Thornton & Ocasio matrix fleshed out into full sentences). The matcher
    returns a weight per logic.

The same questionnaire is applied identically to every lab and source type so
the resulting profiles are directly comparable.
"""

# The seven institutional logics (Thornton & Ocasio's institutional orders).
# Family and Religion are retained deliberately: they should score ~0% for AI
# labs, which serves as a built-in sanity check on the whole method.
LOGICS = ["State", "Profession", "Market", "Corporation", "Family", "Religion", "Community"]

# The nine elemental categories (rows of the matrix). Declared explicitly so
# iteration order is stable and independent of dict insertion details.
CATEGORIES = [
    "Basis of Norms",
    "Sources of Legitimacy",
    "Sources of Authority",
    "Technology Affordances",
    "Sources of Identity",
    "Basis of Attention",
    "Basis of Strategy",
    "Informal Control",
    "Economic System",
]

# Question design principles (kept when rewriting):
#   1. NEVER enumerate the logics or their keywords inside a question — that
#      leads the model toward an option, which is exactly the failure mode
#      answer matching exists to avoid. Questions must be open-ended; the
#      classification happens in the matcher, against the reference answers.
#   2. Ask about CONCRETE, observable things (decisions, funding, access terms,
#      criticism, governance bodies) so the question doubles as a good retrieval
#      query against real documents — not abstract sociological vocabulary.
#   3. The 3 variants per category triangulate: (a) self-description / stated
#      commitments, (b) observable behavior or third-party description,
#      (c) a contested case or trade-off, where the operative logic shows
#      itself most clearly.
QUESTIONNAIRE = {
    "Basis of Norms": {
        "questions": [
            "According to the documents, to whom does {org} say it owes its primary obligations when developing AI, and whose interests does it commit to serving? Cite specific charters, commitments, or statements.",
            "What standards of appropriate conduct is {org} held to in its AI work — by itself and by others — and where do those standards come from?",
            "Describe a case where {org} followed, defended, or broke from common practice in AI development. What obligation or duty did it invoke to explain its conduct?",
        ],
        "reference_answers": {
            "State":       "The lab's obligations are owed to the welfare of the people within a country, the laws of the country, and the legitimate authority in power. Conduct is appropriate when it serves the laws of the country, complies with regulation and bureaucratic orders, and honors commitments made to the state.",
            "Profession":  "Conduct is held to the standards of the scientific and research community: peer review, publication norms, methodological rigor, responsible-disclosure practices, and the shared duties of the AI research field.",
            "Market":      "Appropriate conduct is whatever serves the lab's own competitive self-interest: norms are framed around winning customers, outpacing rivals, and maximizing the lab's own advantage.",
            "Corporation": "Obligations derive from firm membership and employment: people are expected to follow company policy, protect the firm's interests, and act according to internal rules and their role in the organization.",
            "Family":      "Obligation flows from personal loyalty to founders and an inner circle, like a household: standing by one's own people outweighs external standards.",
            "Religion":    "Conduct is governed by devotion to a sacred, transcendent mission: behavior is appropriate when it expresses faith in the cause, with the mission treated like an article of belief rather than a testable claim.",
            "Community":   "Norms come from belonging to a wider movement or community (e.g., open-source or AI-safety communities): appropriate conduct is what good members of that group do for one another.",
        },
    },
    "Sources of Legitimacy": {
        "questions": [
            "When {org} argues it should be trusted to build powerful AI systems, what evidence, credentials, or track record does it offer? Cite specific arguments.",
            "How do {org} and third parties writing about it establish that its approach to AI development and safety is credible? What achievements, qualities, or endorsements get cited?",
            "When {org} has faced public criticism or skepticism about its conduct, on what grounds did it — or its defenders — argue that its actions were justified?",
        ],
        "reference_answers": {
            "State":       "Legitimacy is claimed through serving the state, maintaining social order, and public benefit. It cooperates with regulators, testifying before legislatures, supporting government oversight, and serving the public interest.",
            "Profession":  "Legitimacy rests on technical and scientific expertise: world-class researchers, breakthroughs, peer-reviewed publications, and the authority of expert judgment.",
            "Market":      "Legitimacy comes from market performance: valuation, revenue growth, customer adoption, and investor confidence are what validate the lab's choices.",
            "Corporation": "Legitimacy rests on the firm's standing and market position: being a leading, established company with dominant products is what entitles it to act.",
            "Family":      "Legitimacy rests on unconditional personal loyalty: the founder's blessing and the trust of the inner circle are what make an action right.",
            "Religion":    "Legitimacy flows from the sanctity of the mission itself: the cause is treated as self-justifying and morally transcendent, beyond ordinary evidence or accountability.",
            "Community":   "Legitimacy is earned through trust and reciprocity with a community: openness, transparency, listening to members and users, and a track record of good faith.",
        },
    },
    "Sources of Authority": {
        "questions": [
            "Who makes the final call on {org}'s most consequential AI decisions — for instance whether to train, deploy, pause, or restrict a model? Describe the decision-makers and structures the documents actually name.",
            "What oversight bodies, governance structures, or external actors does {org} answer to in practice, and which of them does it treat as binding rather than advisory?",
            "Describe an instance where a major decision at or about {org} was contested, overridden, or reversed. Who turned out to hold the real power to decide?",
        ],
        "reference_answers": {
            "State":       "Final authority rests with governments and regulators: the lab defers to legal mandates, regulatory approval, national-security review, and state oversight bodies.",
            "Profession":  "Authority rests with expert peers: scientific advisory boards, researcher consensus, and the judgment of acknowledged technical authorities inside and outside the lab.",
            "Market":      "Authority rests with investors and shareholders: funding terms, investor board seats, and shareholder pressure determine what the lab can and cannot do.",
            "Corporation": "Authority rests with executive leadership: the CEO and top management make the final calls, exercised through the corporate hierarchy and board.",
            "Family":      "Authority rests with a patriarch-like founder whose personal will dominates outcomes regardless of formal structures.",
            "Religion":    "Authority rests with charismatic, prophet-like leaders whose vision is followed as revelation rather than evaluated as argument.",
            "Community":   "Authority rests with the community's shared values and ideology: collective norms, member voice, and participatory processes decide.",
        },
    },
    "Technology Affordances": {
        "questions": [
            "What does {org} emphasize that its technology enables people to do? Describe the main capabilities highlighted in its model releases, products, and system documentation.",
            "Who gets access to {org}'s models and infrastructure, on what terms, and what reasons does {org} give for those access decisions?",
            "When {org} describes the purpose of its technical infrastructure — platforms, APIs, evaluation tools, safety systems — what is that infrastructure said to be for?",
        ],
        # NOTE: Family & Religion have no canonical affordance in the literature;
        # these are deliberately weak so they score ~0 (sanity check).
        "reference_answers": {
            # Base State = the Q1 affordance; Q2/Q3 override below.
            "State":       "The technology can enable: 1. efficient coordination and control of the population through the state apparatus and bureaucracy, thereby improving administrative reach; 2. central planning and redistribution efforts; 3. building a national capacity of AI.",
            "Profession":  "Technology serves to enhance knowledgeability and autonomy: research tools, model access for scientists, and capabilities that deepen expert understanding and independent judgment.",
            "Market":      "Technology serves to stimulate and coordinate transactions: commercial products, usage-priced APIs, marketplaces, and features designed to drive exchange between buyers and sellers.",
            "Corporation": "Technology serves to standardize and control operations: enterprise integration, centralized control over deployment and use, and uniform, managed processes.",
            "Family":      "Technology mainly serves to reinforce the inner circle's bonds and position (no canonical affordance in the literature).",
            "Religion":    "Technology mainly serves as an expression of the sacred mission (no canonical affordance in the literature).",
            "Community":   "Technology serves to connect members and open governance: open weights and tooling, community forums and contributions, and participatory decision processes.",
        },
        # State's affordance differs by question; the other six logics are stable.
        "reference_overrides": {
            2: {"State": "Access depends on political capital: the state has meaningful control of any strategic infrastructure and capital, often through licensing, permits, or regulation. The stated reasons are egalitarian."},
            3: {"State": "The justification is framed in the language of national interest, public service, and the collective good."},
        },
    },
    "Sources of Identity": {
        "questions": [
            "How does {org} describe what kind of organization it is, and how do outside observers characterize it? Cite self-descriptions and third-party characterizations.",
            "What achievements or qualities does {org} most prominently claim credit for, and what does it appear to want to be known for?",
            "When {org} distinguishes itself from other AI developers, what differences does it emphasize?",
        ],
        "reference_answers": {
            # Base State = the Q1 identity; Q2/Q3 override below.
            "State":       "The lab identifies as a contributor to a larger collective project — a servant to the nation and the masses. Outside observers characterize it through its political embeddedness, its relationship to the ownership structure, and its alignment with state goals and priorities.",
            "Profession":  "The lab identifies through craft and reputation: a research institution known for the quality of its science and the eminence of its researchers.",
            "Market":      "The lab presents as an interchangeable market actor: one company among competitors, defined by products, pricing, and market share rather than a distinctive persona.",
            "Corporation": "Identity comes from organizational structure and role: the lab describes itself through its divisions, hierarchy, parent company, and corporate position.",
            "Family":      "Identity centers on a founder family or inner circle: the lab is understood as someone's personal project, defined by lineage and insider belonging.",
            "Religion":    "Identity is devotional: the lab presents itself as a congregation united by faith in a transcendent cause.",
            "Community":   "Identity is relational and emotional: the lab presents itself as a movement or community that members and supporters feel they belong to.",
        },
        "reference_overrides": {
            2: {"State": "It claims credit for its contributions to the nation and its economic development — job creation and the provision of welfare (e.g. healthcare). It wishes to be known as a socially responsible and nationally important organization."},
            3: {"State": "It distinguishes itself through attributes that signal greater alignment with collective norms within the state: contribution to the nation's interest and reputation."},
        },
    },
    "Basis of Attention": {
        "questions": [
            "What metrics, milestones, or signals does {org} track and report as the measure of how it is doing? Cite specifics from the documents.",
            "Whose reactions does {org} appear most responsive to? Which audiences does it address first and most carefully in its communications and decisions?",
            "What kinds of developments or events have prompted {org} to change its behavior — for example competitor releases, new regulation, research findings, or public reaction?",
        ],
        "reference_answers": {
            # The source gives State only for Q2 and Q3. Per design, Q1 reuses the
            # category's other text, so base State = the Q2 exemplar; Q3 overrides.
            "State":       "Attention tracks entities with public and governmental standing — ministries, proposed regulations, and the officials responsible for granting capital and data access. Communication addresses alignment with national development and public benefit; the secondary audience is the public, reassuring on safety, stability, and jobs without destabilizing disruption.",
            "Profession":  "Attention tracks professional standing: benchmark results, publications and citations, research breakthroughs, and esteem among scientific peers.",
            "Market":      "Attention tracks market position: market share, competitor moves, revenue, and product adoption are the signals that drive responses.",
            "Corporation": "Attention tracks the internal hierarchy: reorganizations, leadership changes, reporting lines, and standing within the firm.",
            "Family":      "Attention tracks the inner circle: who is in or out of favor with the founder and the household around them.",
            "Religion":    "Attention tracks fidelity to the mission: perceived progress toward the transcendent goal eclipses worldly metrics.",
            "Community":   "Attention tracks member engagement: community contributions, participation, and how invested members are in the group.",
        },
        "reference_overrides": {
            3: {"State": "Behavior change is triggered by actions of the state — new regulation, content and security directives, and shifts in national-plan priorities. Another trigger is anything that threatens political or social stability or the firm's legitimacy with the state: public grievances, social-movement attention, or media coverage."},
        },
    },
    "Basis of Strategy": {
        "questions": [
            "What does {org} say it is ultimately trying to achieve with AI, and what intermediate goals structure its roadmap? Cite stated objectives.",
            "Describe a major strategic choice {org} made — a partnership, product launch, restructuring, or research bet. What payoff was that choice said to pursue?",
            "When {org} discusses trade-offs between safety and its other goals, how does it describe the trade-off and what does it say takes priority?",
        ],
        "reference_answers": {
            "State":       "Strategy aims to increase the public good: choices are justified by societal benefit, safety for humanity, and broad, equitable distribution of AI's gains.",
            "Profession":  "Strategy aims to increase reputation for excellence: research leadership, scientific firsts, and recognition within the field.",
            "Market":      "Strategy aims to increase efficiency and profit: revenue growth, cost advantage, and winning commercially against rivals.",
            "Corporation": "Strategy aims to increase the organization's size and diversification: expanding headcount and scope, entering new business lines, and consolidating the firm's reach.",
            "Family":      "Strategy aims to increase the honor, security, and succession of the founder's inner circle.",
            "Religion":    "Strategy aims to advance the sacred mission symbolically, even at material cost.",
            "Community":   "Strategy aims to raise the status and honor of members: empowering the community and rewarding its contributors.",
        },
    },
    "Informal Control": {
        "questions": [
            "Apart from formal rules and contracts, what keeps people at {org} acting in line with its expectations? What social pressures, cultural norms, or reputational forces do the documents describe?",
            "How are internal dissent, departures, and whistleblowing handled at or around {org}? What has happened to insiders who deviated from expected behavior?",
            "What role do outside watchers — peers, analysts, journalists, online communities — play in checking or disciplining {org}'s behavior?",
        ],
        "reference_answers": {
            "State":       "Discipline operates through political maneuvering: lobbying, alliances, and backroom negotiation with government and interest-group actors.",
            "Profession":  "Discipline operates through celebrated experts: the esteem or censure of renowned researchers and scientific peers keeps behavior in line.",
            "Market":      "Discipline operates through analysts and market scrutiny: press coverage of performance, investor reactions, and industry rankings.",
            "Corporation": "Discipline operates through organizational culture: internal values, socialization, and 'the way we do things here.'",
            "Family":      "Discipline operates through household politics: personal favor, loyalty tests, and intrigue within the inner circle.",
            "Religion":    "Discipline operates through devotion: a felt calling and the fear of betraying the faith keep members in line.",
            "Community":   "Discipline operates through visibility: actions are exposed to the community, and reputation among members enforces the norms.",
        },
    },
    "Economic System": {
        "questions": [
            "How is {org} funded and how does it sustain its operations? Describe its revenue sources, investors, ownership, and corporate structure as given in the documents.",
            "How does {org} explain the relationship between its mission and its commercial activities — why does it say it makes money the way it does?",
            "Who captures the economic value {org} creates, and how are ownership, profit, and control distributed?",
        ],
        "reference_answers": {
            "State":       "Welfare capitalism: funding and structure are oriented to public benefit — capped returns, nonprofit oversight, public-interest missions, or partnership with the state.",
            "Profession":  "Personal capitalism: the enterprise is sustained by the expertise and reputation of its practitioners, who are its central economic asset.",
            "Market":      "Market capitalism: revenue comes from competitive sale of products and services, disciplined by price, demand, and competition.",
            "Corporation": "Managerial capitalism: a professionally managed firm whose hierarchy allocates capital, pursues scale, and reinvests for growth.",
            "Family":      "Family capitalism: ownership and control are concentrated in a founder family or personal inner circle.",
            "Religion":    "Vocational (occidental) capitalism: economic activity is treated as the dutiful expression of a calling rather than a means to profit.",
            "Community":   "Cooperative capitalism: value is shared among members — open assets, commons-based production, and cooperative or distributed ownership.",
        },
    },
}


def build_questionnaire(org: str) -> list[dict]:
    """Materialize the questionnaire for one lab.

    Returns one dict per question:
        qid       stable identifier, e.g. "Basis of Norms#2"
        category  the elemental category the question probes
        variant   1-based phrasing index within the category
        question  the org-templated question text
    """
    items = []
    for category in CATEGORIES:
        for variant, template in enumerate(QUESTIONNAIRE[category]["questions"], start=1):
            items.append({
                "qid": f"{category}#{variant}",
                "category": category,
                "variant": variant,
                "question": template.format(org=org),
            })
    return items


def reference_answers(category: str, variant: int | None = None) -> dict[str, str]:
    """The 7 per-logic reference answers for a (category, variant) — the matching space.

    The category's `reference_answers` block is the base set shared by all three
    question variants. If the category declares `reference_overrides`
    ({variant: {logic: text}}), those cells replace the base for that variant;
    variants without an override reuse the base text. Passing variant=None
    returns the base set unchanged.
    """
    refs = dict(QUESTIONNAIRE[category]["reference_answers"])
    if variant is not None:
        overrides = QUESTIONNAIRE[category].get("reference_overrides", {})
        refs.update(overrides.get(variant, {}))
    return refs


# Structural invariants the scoring pipeline depends on. Every category must have
# 3 questions and a full 7-logic base reference set; any overrides must target a
# real variant (1-3) and known logics, so every (category, variant) resolves to
# all 7 logics.
assert set(QUESTIONNAIRE) == set(CATEGORIES), "category set mismatch"
for _cat, _block in QUESTIONNAIRE.items():
    assert len(_block["questions"]) == 3, f"{_cat}: expected 3 questions"
    assert set(_block["reference_answers"]) == set(LOGICS), f"{_cat}: expected 7 reference answers"
    for _v, _ov in _block.get("reference_overrides", {}).items():
        assert _v in (1, 2, 3), f"{_cat}: override variant {_v} out of range"
        assert set(_ov) <= set(LOGICS), f"{_cat}: override targets an unknown logic"

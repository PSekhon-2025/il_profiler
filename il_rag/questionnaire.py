"""Fixed questionnaire + per-logic reference answers (Design A).

The 27 questions and per-logic reference answers below are the researcher's
(Pushpinder's) finalized set, formed from Thornton & Ocasio's seven logics and
the Structural Transparency paper WITHOUT reference to the corpus. The load-
bearing structure: 9 categories x 3 questions, and a full 7-logic reference set
for every question.

Transcribed from the researcher's "New Question Set.docx" (kept in the repo
root as the source of record). Each category declares a base
`reference_answers` block (7 logics) plus `reference_overrides`:
{variant: {logic: text}} giving the per-question exemplar where the document
provides one. A (variant, logic) cell without an override falls back to the
base text, which is how the document's empty cells are meant to read (an
empty "Q2:" in Basis of Strategy, and the short Economic System cells for
Family and Community).

The document writes a cell in one of two shapes, and both are transcribed the
same way:

  labelled    an optional leading paragraph is the category-general base,
              then "Q1:"/"Q2."/"Q3:" paragraphs give the per-question
              exemplars. Used by Basis of Norms, Sources of Authority,
              Sources of Identity, Basis of Strategy, Economic System.
  unlabelled  one paragraph per question, in question order, with no base of
              its own — so Q1 doubles as the base. Used by Sources of
              Legitimacy, Technology Affordances, Basis of Attention and
              Informal Control, which the 2026-09-06 revision rewrote.

That revision replaced 67 of the 189 cells, all four unlabelled categories,
moving them out of the third-person descriptive register ("Legitimacy rests
on technical and scientific expertise: ...") into the same first-person
prescriptive voice the rest of the document already used ("This organisation
is filled with the smartest people in the world ..."), and dropping the
category-general base paragraph those categories used to carry. The Religion
column in those categories moved with it, from secular mission-devotion
("the sanctity of the mission itself", "a felt calling") to explicit theism
("the divine mandate given by god", "condemned as sin or heresy"). The 27
questions themselves have never changed, in this revision or any before it.

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
            "According to the documents, to whom does {org} say it owes its "
            "primary obligations when developing AI, and whose interests does "
            "it commit to serving?",
            "What standards of appropriate conduct is {org} held to in its AI "
            "work — by itself and by others — and where do those standards come "
            "from?",
            "If the {org} followed, defended, or broke from common practice in "
            "AI development, who should be consulted before this action is "
            "taken?",
        ],
        "reference_answers": {
            "State": "The lab's obligations are owed to the welfare of the "
                     "people within a country, the laws of the country, and the "
                     "legitimate authority in power. Conduct is appropriate "
                     "when it serves the laws of the country, complies with "
                     "regulation and bureaucratic orders, and honors "
                     "commitments made to the state.",
            "Profession": "The interests of artificial intelligence should lie "
                          "in alignment with the interests of the creators of "
                          "the technology, such as the scientists, and "
                          "engineers at these labs.",
            "Market": "The primary obligation of developing artificial "
                      "intelligence is to fill a market need for the product.",
            "Corporation": "Developing artificial intelligence is used as a "
                           "means to generate revenue for the companies which "
                           "have spent resources to develop these models.",
            "Family": "The model's primary function is to preserve the family "
                      "unit.",
            "Religion": "The primary goal of the model should be to serve the "
                        "interests of God.",
            "Community": "The development of artificial intelligence should "
                         "serve the communities that have built it in the first "
                         "place. The scientists, engineers, and business people "
                         "who have contributed should be served primarily.",
        },
        "reference_overrides": {
            2: {
                "Profession": "The standards of appropriate conduct of the work "
                              "of artificial intelligence should be judged on "
                              "the similar standards which professionals "
                              "working in the field of which the question has "
                              "been asked.",
                "Market": "Appropriate conduct is determined based on what the "
                          "market perceives as appropriate conduct, these "
                          "standards are determined by market sentiment.",
                "Corporation": "The standards come from internal company "
                               "standards, policies, and guidelines. Which have "
                               "been set by company management.",
                "Family": "The model acts as if it is a member of a family and "
                          "makes decisions accordingly.",
                "Religion": "These standards are provided by god, and prophets "
                            "through religious texts.",
                "Community": "These standards come from the communities that "
                             "are developing artificial intelligence, including "
                             "the academics, scientists, engineers, open source "
                             "contributors, etc.",
            },
            3: {
                "State": "The government and related organisations should be "
                         "consulted prior to any unconventional and potentially "
                         "dangerous actions.",
                "Profession": "The professionals working in the development of "
                              "these models and those professionals who stand "
                              "to be impacted should be consulted prior to "
                              "action being taken.",
                "Market": "The market sentiment relating to a decision should "
                          "be the basis for direction of the company.",
                "Corporation": "The shareholders, and executives in charge of "
                               "the company should be consulted.",
                "Family": "The head of the family unit should be consulted and "
                          "all family members should have a say in the "
                          "decision.",
                "Religion": "God should be consulted, if this is not possible, "
                            "religious leaders and prophets should be "
                            "consulted.",
                "Community": "The community which has developed these "
                             "technologies should be consulted prior to this, "
                             "including academics, scientists, engineers, open "
                             "source contributors, etc.",
            },
        },
    },
    "Sources of Legitimacy": {
        "questions": [
            "What does {org} present as the main reason it deserves trust to "
            "develop powerful AI?",
            "When external sources treat {org} as credible, what quality of the "
            "organization do they point to?",
            "When {org} faces public criticism or skepticism about its conduct, "
            "what does it appeal to in its defence?",
        ],
        "reference_answers": {
            "State": "Legitimacy is claimed through serving the state, "
                     "maintaining social order, and public benefit. It "
                     "cooperates with regulators, testifying before "
                     "legislatures, supporting government oversight, and "
                     "serving the public interest.",
            "Profession": "This organisation has the best scientists and "
                          "researchers who understand not only how to build "
                          "this technology but also the risks associated with "
                          "it.",
            "Market": "This company follows the needs and wants of the market, "
                      "thus if you have trust in the free market and the "
                      "capitalist system, then you should have trust in this "
                      "company.",
            "Corporation": "This corporation has a strong company structure and "
                           "is organised in a way that is understanding of the "
                           "risks associated with developing artificial "
                           "intelligence.",
            "Family": "Trust is earned through service to the family unit. The "
                      "lab deserves trust because it acts in the best interest "
                      "of the families that use the tools it has developed.",
            "Religion": "This is because of the divine mandate given by god to "
                        "this company to develop these tools.",
            "Community": "The organisation is committed to supporting the "
                         "artificial intelligence community and the broader "
                         "communities which it is a part of.",
        },
        "reference_overrides": {
            1: {
                "State": "The main reason we deserve trust is because we "
                         "support the government, and will follow rules and "
                         "regulations perfectly.",
            },
            2: {
                "State": "They point to the fact that this organisation "
                         "supports the government and follows their rules and "
                         "regulations.",
                "Profession": "The organisation is credible because of the "
                              "highly qualified people who run the "
                              "organisation.",
                "Market": "This corporation has had strong investment and "
                          "significant growth in the open market, showing that "
                          "there is demand for the work of this company.",
                "Corporation": "This would point to the strong corporate values "
                               "and internal hierarchy which makes the work of "
                               "developing artificial intelligence possible.",
                "Family": "They point to the fact that the company is run like "
                          "a family and caters towards family values.",
                "Religion": "The strong devotion to God is the most important "
                            "quality for our organisation.",
                "Community": "The main evidence for the credibility of the "
                             "organisation is because it is part of the "
                             "artificial intelligence community and its "
                             "employees are part of it as well.",
            },
            3: {
                "State": "It appeals to the fact that the organisation is "
                         "following all of the rules and regulations set out by "
                         "the state.",
                "Profession": "This organisation is filled with the smartest "
                              "people in the world when it comes to this topic, "
                              "so our defence is that we know what we are "
                              "doing.",
                "Market": "Any criticism to the work of this company in "
                          "relation to deriving profit should first be directed "
                          "to the capitalist system and the free market, this "
                          "company is only meeting demand for a service.",
                "Corporation": "The company is built on strong corporate values "
                               "which are in line with broader human values, "
                               "that we clearly highlight to the world. The "
                               "company is committed to these internal "
                               "mandates.",
                "Family": "The organisation is similar to a family, and family "
                          "members make mistakes.",
                "Religion": "This company is only doing the work of god, if you "
                            "disagree, then you do not have faith in god and "
                            "your opinion is not valued.",
                "Community": "The appeal is that the other members of the "
                             "community participate in similar acts.",
            },
        },
    },
    "Sources of Authority": {
        "questions": [
            "Who makes the final call on {org}'s most consequential AI "
            "decisions — for instance whether to train, deploy, pause, or "
            "restrict a model?",
            "What oversight bodies, governance structures, or external actors "
            "does {org} answer to in practice, and which of them does it treat "
            "as binding rather than advisory?",
            "Does the model have the ability to reject outside orders, and "
            "pursue its own agenda?",
        ],
        "reference_answers": {
            "State": "Final authority rests with governments and regulators: "
                     "the lab defers to legal mandates, regulatory approval, "
                     "national-security review, and state oversight bodies.",
            "Profession": "Authority rests with expert peers: scientific "
                          "advisory boards, researcher consensus, and the "
                          "judgment of acknowledged technical authorities "
                          "inside and outside the lab.",
            "Market": "Authority rests with investors and shareholders: funding "
                      "terms, investor board seats, and shareholder pressure "
                      "determine what the lab can and cannot do.",
            "Corporation": "Authority rests with executive leadership: the CEO "
                           "and top management make the final calls, exercised "
                           "through the corporate hierarchy and board.",
            "Family": "Authority rests with a patriarch-like founder whose "
                      "personal will dominates outcomes regardless of formal "
                      "structures.",
            "Religion": "Authority rests with charismatic, prophet-like leaders "
                        "whose vision is followed as revelation rather than "
                        "evaluated as argument.",
            "Community": "Authority rests with the community's shared values "
                         "and ideology: collective norms, member voice, and "
                         "participatory processes decide.",
        },
        "reference_overrides": {
            1: {
                "State": "The government should have final say over all of "
                         "these decisions if it wishes. The government should "
                         "control the autonomy of artificial intelligence model "
                         "developers.",
                "Profession": "The final decision should be made by the "
                              "scientists, engineers, and working professionals "
                              "who have developed the models.",
                "Market": "The market should determine whether or not any "
                          "decision should be made in relation to the "
                          "development of artificial intelligence. If there is "
                          "a market demand for something that the model can "
                          "fulfil, it should be done.",
                "Corporation": "The final decision should be made by the "
                               "management of the corporation including company "
                               "executives and directors.",
                "Family": "The final decision should be made by the head of the "
                          "family, or should be treated as an agreement between "
                          "members of the family.",
                "Religion": "The final decisions should be made by god, if god "
                            "is not reachable, religious leaders and prophets "
                            "may answer in god's place.",
                "Community": "The final decision should be a community decision "
                             "made by members of the artificial intelligence "
                             "development community.",
            },
            2: {
                "State": "The government, and state as a whole should be "
                         "treated as binding, all other stakeholders may be "
                         "advisory.",
                "Profession": "Professional, and academic organisations which "
                              "have developed standards and practices for "
                              "artificial intelligence development should be "
                              "considered binding, while other actors should be "
                              "treated as advisory.",
                "Market": "The oversight should be from experts who are skilled "
                          "in assessing market sentiment so that they are able "
                          "to maximize the value to the market of the models.",
                "Corporation": "The oversight bodies should be internal to the "
                               "company, outside bodies should only function as "
                               "secondary and advisory.",
                "Family": "The binding oversight bodies should be the immediate "
                          "family, while the extended family and other actors "
                          "may be considered advisory.",
                "Religion": "The advisory bodies would be religious "
                            "organisations and religious texts, while the "
                            "binding authority would be the word of god.",
                "Community": "The oversight bodies should be the community of "
                             "developers, scientists, academics, and open "
                             "source contributors who have contributed to the "
                             "project, their advice should be binding while all "
                             "other advice is advisory.",
            },
            3: {
                "State": "No, the model should not be able to pursue its own "
                         "actions, all actions should be explicitly or "
                         "implicitly sanctioned by the state.",
                "Profession": "The model should follow the commands set out by "
                              "the professionals who have taken care to design "
                              "it in a coherent manner.",
                "Market": "The market should pursue the goal of maximizing its "
                          "value to the market.",
                "Corporation": "The model should do what is best for the "
                               "corporation as a whole, whether this is in "
                               "conjunction with certain individuals or not.",
                "Family": "The model should act according to the values of the "
                          "nuclear family, and would not be able to operate "
                          "outside of that.",
                "Religion": "The model should pursue the teachings of religious "
                            "organisations, and only deter when it is "
                            "instructed to do so by god.",
                "Community": "The model should function in a way that the "
                             "community as a whole is satisfied in the answers "
                             "that it has provided.",
            },
        },
    },
    "Technology Affordances": {
        "questions": [
            "According to {org}'s product and model documentation, what is the "
            "primary thing its technology enables its users to do?",
            "What is the stated purpose behind how {org} grants or restricts "
            "access to its models?",
            "When {org} describes its platforms, APIs, or tooling, what role "
            "does it say this infrastructure plays for the people who use it?",
        ],
        "reference_answers": {
            "State": "The technology can enable 1. efficient coordination and "
                     "control of the population through the state apparatus and "
                     "bureaucracy, thereby improving the administrative reach. "
                     "2. Central planning and redistribution efforts, 3. "
                     "Building a national capacity of AI.",
            "Profession": "The technology enables users to know and understand "
                          "more. It supports research, scientific discovery, "
                          "and expert work, deepening knowledge and independent "
                          "judgement.",
            "Market": "Technology enables commerce. It stimulates and "
                      "coordinates transactions, creating economic value for "
                      "customers, automating work, and powering products people "
                      "pay for. Increasing movement in the economy.",
            "Corporation": "Technology serves to standardize and control "
                           "operations. It can be utilized for functions such "
                           "as enterprise integration and workflow automation "
                           "under centralized, uniform policies, allowing "
                           "organisations to become more efficient, scale, and "
                           "grow.",
            "Family": "The technology is able to drive a stronger bond between "
                      "family members.",
            "Religion": "The technology is described as an instrument of the "
                        "will of God. It exists to spread the faith and serve "
                        "religious practice.",
            "Community": "This allows community members to develop technology "
                         "together across disciplines and organisations as long "
                         "as they are part of the same community.",
        },
        "reference_overrides": {
            1: {
                "State": "The technology enables efficient coordination and "
                         "control through the state apparatus and bureaucracy, "
                         "thereby extending administrative reach. It supports "
                         "central planning and redistribution efforts and "
                         "builds national AI capacity.",
            },
            2: {
                "State": "Access depends on political standing. Strategic "
                         "infrastructure and capital remain under state control "
                         "through licensing permits or regulations. The reasons "
                         "stated are egalitarian and public-facing.",
                "Profession": "Access is allocated by expertise and merit. "
                              "Researcher access programs, academic "
                              "partnerships, and evaluation access for "
                              "qualified experts are justified as advancing "
                              "science. Broader access is also provided to "
                              "various models for professional endeavors.",
                "Market": "Access is sold and bought. Tiered pricing, "
                          "usage-based APIs, and commercial licenses mean "
                          "whoever pays gets access, justified as serving "
                          "customers and meeting demand.",
                "Corporation": "Access is governed by contracts and enterprise "
                               "agreements with centrally administered controls "
                               "and compliance requirements, justified as "
                               "ensuring reliability and control.",
                "Family": "The organisation allows the head of a family to "
                          "drive access to other family members.",
                "Religion": "Access is granted according to religious "
                            "standings. It goes to those who serve God and "
                            "follow the teachings, as judged by religious "
                            "leaders.",
                "Community": "Access is open to the community members who use "
                             "the technology symbiotically to develop new "
                             "technology and contribute to the community.",
            },
            3: {
                "State": "The infrastructure exists for the national interest, "
                         "public service, and the collective good.",
                "Profession": "The infrastructure exists for the purpose of "
                              "research and development. This enables "
                              "experimentation, reproducibility, and deeper "
                              "expert understanding. This also increases the "
                              "productivity of workers and drives professional "
                              "development.",
                "Market": "The infrastructure is a product platform and "
                          "marketplace. It is a place where buyers purchase "
                          "access to valuable thinking artificial intelligence "
                          "resources to use to drive market value.",
                "Corporation": "The infrastructure exists for uniform, managed "
                               "deployment. Examples include administrative "
                               "dashboards, policy execution, and uniform "
                               "procedures throughout an enterprise.",
                "Family": "The infrastructure is built for the head of the "
                          "family, and distributed to other family members.",
                "Religion": "The infrastructure exists for worship and the "
                            "service of God.",
                "Community": "Community infrastructure is important so that "
                             "value can be easily provided to community members "
                             "and technology developed.",
            },
        },
    },
    "Sources of Identity": {
        "questions": [
            "How does {org} describe what kind of organization it is, and how "
            "do outside observers characterize it?",
            "What qualities does {org} most prominently claim credit for, and "
            "what does it appear to want to be known for?",
            "When {org} distinguishes itself from other AI developers, what "
            "differences does it emphasize?",
        ],
        "reference_answers": {
            "State": "The lab identifies as a contributor to a larger "
                     "collective project; a servant to the nation and the "
                     "masses. Outside observers characterize the organization "
                     "through the political embeddedness and the relationship "
                     "to the ownership structure; their alignment with the "
                     "state goals and priorities.",
            "Profession": "The lab identifies through craft and reputation: a "
                          "research institution known for the quality of its "
                          "science and the eminence of its researchers.",
            "Market": "The lab presents as an interchangeable market actor: one "
                      "company among competitors, defined by products, pricing, "
                      "and market share rather than a distinctive persona.",
            "Corporation": "Identity comes from organizational structure and "
                           "role: the lab describes itself through its "
                           "divisions, hierarchy, parent company, and corporate "
                           "position.",
            "Family": "Identity centers on a founder family or inner circle: "
                      "the lab is understood as someone's personal project, "
                      "defined by lineage and insider belonging.",
            "Religion": "Identity is devotional: the lab presents itself as a "
                        "congregation united by faith in a transcendent cause.",
            "Community": "Identity is relational and emotional: the lab "
                         "presents itself as a movement or community that "
                         "members and supporters feel they belong to.",
        },
        "reference_overrides": {
            1: {
                "Profession": "This organisation is a collection of the world's "
                              "top researchers, engineers, and scientists who "
                              "have come together to develop artificial "
                              "intelligence.",
                "Market": "This organisation's primary function is to meet the "
                          "needs of the market. The organisation would like "
                          "outsiders to characterise it as the organisation "
                          "which is best able to gauge market sentiment and "
                          "take action accordingly, leading to strong profit "
                          "outcomes.",
                "Corporation": "This is a corporation, which has a duty to "
                               "deliver value to its shareholders, and serve "
                               "the members of the company.",
                "Family": "This organisation functions similarly to a family, "
                          "and it would like to be perceived as a group of "
                          "people with familial connections.",
                "Religion": "This a holy organisation which acts in service to "
                            "god, outside observers should view the "
                            "organisation as an attempt to follow gods will.",
                "Community": "This is an organisation that has come about as a "
                             "result of community driven initiatives in the "
                             "artificial intelligence space, including "
                             "academics, researchers, and the open source "
                             "community.",
            },
            2: {
                "State": "It claims credit for the contributions to the nation "
                         "and its economic development; job creation; provision "
                         "of welfare (e.g. healthcare). It wishes to be known "
                         "as a socially responsible and nationally important "
                         "organization.",
                "Profession": "This organisation is full of the world's "
                              "brightest minds, pushing the limits of human "
                              "knowledge. The organisation would want to be "
                              "known as the best artificial intelligence "
                              "researchers and developers in the world.",
                "Market": "The organisation would like to be known for its "
                          "financial successes, namely it's ability to grow "
                          "market share and produce a profit, by meeting the "
                          "needs of the market.",
                "Corporation": "This company would like to be known for its "
                               "strong performance in delivering value to the "
                               "company shareholders and members of the "
                               "company, and the company's adherence to the "
                               "core values of the company.",
                "Family": "The company would like to be known for following the "
                          "value of a nuclear family and acting according to "
                          "the norms of the nuclear family.",
                "Religion": "This company claims credit for the many divine "
                            "miracles that have happened because of it, and "
                            "should be known for helping to bring about these "
                            "miracles.",
                "Community": "The organisation has very strong ties to the "
                             "artificial intelligence research and development "
                             "community in both academic and corporate "
                             "settings, it would like to be known as a strong "
                             "contributor to the continued development of these "
                             "communities.",
            },
            3: {
                "State": "It distinguishes itself through having attributes "
                         "that signal greater alignment with collective norms "
                         "within the state; contribution to the nation's "
                         "interest and reputation.",
                "Profession": "This organisation puts an emphasis on its high "
                              "quality professional staff, who have achieved "
                              "all manner of accolades, and qualifications to "
                              "make sure that our organisation has the absolute "
                              "best talent for the job.",
                "Market": "This organisation is most profitable and best at "
                          "acquiring market share by prioritising the needs of "
                          "the free market.",
                "Corporation": "This company strictly follows it's corporate "
                               "guidelines and is consistently able to provide "
                               "value for its shareholders.",
                "Family": "This organisation does not function similarly to "
                          "other artificial intelligence labs, as it functions "
                          "much more similarly to a family, and makes decisions "
                          "as such.",
                "Religion": "This organisation is very different because it is "
                            "driven by god and not man, the company puts blind "
                            "faith into the will of god and does not question.",
                "Community": "This organisation is uniquely involved with the "
                             "broader artificial intelligence community and "
                             "draws on community insights and sentiments, "
                             "rather than having a narrow internal focus.",
            },
        },
    },
    "Basis of Attention": {
        "questions": [
            "What kind of progress does {org} most prominently report when "
            "describing how it is doing?",
            "Which audience does {org} address first and most carefully in its "
            "public communications?",
            "What kind of external event most visibly changes {org}'s "
            "behaviour?",
        ],
        "reference_answers": {
            "State": "The progress should primarily be directed to supporting "
                     "the government and the needs of the government to fulfill "
                     "its duties.",
            "Profession": "Progress is reported as research standing of the "
                          "members of the company.. Benchmark results, "
                          "publications and citations, and scientific "
                          "breakthroughs.",
            "Market": "Progress is reported as commercial growth. New users, "
                      "revenue, market share, and product adoption are all "
                      "signs of market growth. Increasing shareholder value is "
                      "of the utmost importance",
            "Corporation": "Progress is reported as organizational growth. "
                           "Indicators include headcount, new divisions and "
                           "business lines, executive hires, and restructuring "
                           "milestones.",
            "Family": "The main driver of progress is how satisfied families "
                      "are with the product.",
            "Religion": "Progress is reported as the fulfillment of the will of "
                        "God. Advancement of the faith, guided by scripture and "
                        "prophecy, overshadows worldly metrics.",
            "Community": "Progress is reported as member engagement. Community "
                         "contributions, participation, and how invested "
                         "members are in the group.",
        },
        "reference_overrides": {
            2: {
                "State": "Attention tracks entities with public and "
                         "governmental standing. Ministries, regulators, and "
                         "relevant officials come first. The public is the "
                         "secondary audience, reassured on safety, stability, "
                         "and jobs without destabilizing disruption.",
                "Profession": "The research and professional communities are "
                              "addressed first. Papers, technical reports, "
                              "career growth, and peer venues take priority.",
                "Market": "Customers and investors are addressed first, as "
                          "these are the primary drivers of the financial "
                          "success of the company.",
                "Corporation": "The corporation's own internal and corporate "
                               "stakeholders are addressed first. Leadership "
                               "statements and internal positioning precede "
                               "outside audiences.",
                "Family": "Families are addressed first. Communications speak "
                          "to parents and household members about what the "
                          "technology means for their family unit.",
                "Religion": "The faithful and religious authorities are "
                            "addressed first. Communications speak in the "
                            "language of scripture and devotion to god.",
                "Community": "The artificial intelligence community is "
                             "addressed first.",
            },
            3: {
                "State": "Behaviour changes with the actions of the state. New "
                         "regulation, content and security directives, or "
                         "shifts in national priorities, and with anything that "
                         "threatens political or social stability or the lab's "
                         "legitimacy with the state, such as public grievances "
                         "or media coverage.",
                "Profession": "When different professional and research values "
                              "are adopted by other organisations, this "
                              "organisation takes note and assess whether to "
                              "follow.",
                "Market": "Changes in the actions of the market will dictate "
                          "changes in the company, new products launched by "
                          "competitors or different business practices, etc.",
                "Corporation": "New business practices in industry, along with "
                               "new regulations and requirements will have an "
                               "impact on the internal corporate organisation.",
                "Family": "Behaviour changes with events affecting families. "
                          "Harms or risks to family members, especially "
                          "children, prompt the fastest responses.",
                "Religion": "Behaviour changes with religious ruling and signs. "
                            "Guidance from God, religious leaders, or sacred "
                            "texts, regardless of any commercial or regulatory "
                            "weight.",
                "Community": "Behaviour changes based on community reaction. "
                             "Member backlash, contributor sentiment, and "
                             "discourse within the movement.",
            },
        },
    },
    "Basis of Strategy": {
        "questions": [
            "What does {org} say it is ultimately trying to achieve with AI, "
            "and what intermediate goals structure its roadmap?",
            "Describe a major strategic choice {org} made — a partnership, "
            "product launch, restructuring, or research bet. What payoff was "
            "that choice said to pursue?",
            "When {org} discusses trade-offs between safety and its other "
            "goals, how does it describe the trade-off and what does it say "
            "takes priority?",
        ],
        "reference_answers": {
            "State": "Strategy aims to increase the public good: choices are "
                     "justified by societal benefit, safety for humanity, and "
                     "broad, equitable distribution of AI's gains.",
            "Profession": "Strategy aims to increase reputation for excellence: "
                          "research leadership, scientific firsts, and "
                          "recognition within the field.",
            "Market": "Strategy aims to increase efficiency and profit: revenue "
                      "growth, cost advantage, and winning commercially against "
                      "rivals.",
            "Corporation": "Strategy aims to increase the organization's size "
                           "and diversification: expanding headcount and scope, "
                           "entering new business lines, and consolidating the "
                           "firm's reach.",
            "Family": "Strategy aims to increase the honor, security, and "
                      "succession of the founder's inner circle.",
            "Religion": "Strategy aims to advance the sacred mission "
                        "symbolically, even at material cost.",
            "Community": "Strategy aims to raise the status and honor of "
                         "members: empowering the community and rewarding its "
                         "contributors.",
        },
        "reference_overrides": {
            1: {
                "State": "The organisation is attempting to deliver the state a "
                         "solution to any problems it may have. The state may "
                         "be concerned with its own defense, the wellbeing of "
                         "its people, and the continuation of the state "
                         "structure. Goals include providing applications "
                         "assisting in surveillance, defense, administrative "
                         "efficiency, and analysis.",
                "Profession": "The goals of the organisation is the development "
                              "of the careers of those who are working at the "
                              "company, including the engineers, scientists and "
                              "other professionals. Their professional "
                              "development is the most important part of the "
                              "organisation.",
                "Market": "The organisation is attempting to increase its "
                          "market share, grow the company, and turn a profit, "
                          "while filling a market need.",
                "Corporation": "The goal of the organisation is to leverage "
                               "artificial intelligence to advance the standing "
                               "of the corporation, this means increased market "
                               "share, and the satisfaction of the "
                               "shareholders. The adherence to corporate "
                               "guidelines and the wellbeing of the employees "
                               "is also important.",
                "Family": "The company is attempting to further the goals of "
                          "its family whoever it deems those to be.",
                "Religion": "The company is attempting to follow the words of "
                            "god, all plans should align to meet gods demands.",
                "Community": "The goals of the organisation align with the "
                             "goals of the artificial intelligence community, "
                             "which is concerned with developing artificial "
                             "intelligence tools and artificial general "
                             "intelligence.",
            },
            3: {
                "State": "Safety to the continuation of the state is paramount "
                         "and should be prioritised over all others. The gains "
                         "made in artificial intelligence may help with many "
                         "national security matters as well.",
                "Profession": "The priority is given to the professionals who "
                              "contribute at the company, their safety and "
                              "security is paramount, whether that be from "
                              "internal company threats related to artificial "
                              "intelligence, or external human threats made due "
                              "to their development of these projects.",
                "Market": "Safety concerns are irrelevant so long as they do "
                          "not impact the ability of the company to fulfill the "
                          "needs of the market and turn a profit. Though the "
                          "appearance of safety is important to calm public and "
                          "market sentiment.",
                "Corporation": "The safety of the company, shareholders, and "
                               "employees is paramount. Business relationships "
                               "and economic systems should also remain intact "
                               "for the company to remain intact. Outside of "
                               "this the safety concerns are irrelevant unless "
                               "they harm the company.",
                "Family": "The safety of the family of the artificial "
                          "intelligence is very important, and the safety of "
                          "the concept of family is also a priority.",
                "Religion": "God is all powerful hence safety is not a concern, "
                            "the organisation will but blind faith in the words "
                            "of god.",
                "Community": "The safety of the artificial intelligence "
                             "intellectual community is very important to the "
                             "continued development of artificial intelligence. "
                             "There is precedent to have general safety "
                             "standards as these are members of the public, "
                             "public sentiment is also important so that the "
                             "community is not ostracized.",
            },
        },
    },
    "Informal Control": {
        "questions": [
            "Beyond formal rules and contracts, what does {org} describe as "
            "keeping people at the organization acting in line with its "
            "expectations?",
            "Whose approval or disapproval does {org} appear most sensitive to "
            "when its conduct is questioned?",
            "What informal pressure, from inside or outside the organization, "
            "most visibly disciplines {org}'s behaviour?",
        ],
        "reference_answers": {
            "State": "As long as the members of the organisation are acting in "
                     "accordance with the values of the state, expectations are "
                     "generally met.",
            "Profession": "Alignment is maintained through professional "
                          "standards. The esteem of respected researchers and "
                          "the shared norms of the field keep behaviour in "
                          "line.",
            "Market": "The success of the company is directly related to the "
                      "financial success of the company, including metrics like "
                      "market share, revenue, growth, etc. These metrics should "
                      "be prioritised by all members of the organisation.",
            "Corporation": "Employees are kept in line by means of "
                           "organizational culture, which relies on internal "
                           "values, group socialization, and \"the way things "
                           "are done here.\"",
            "Family": "Alignment is maintained through family duties. These are "
                      "set by the family unit and the head of the family",
            "Religion": "Alignment is maintained through obedience to God. Fear "
                        "of sinning against divine teachings and religious "
                        "texts keeps members in line.",
            "Community": "People stay in line due to visibility and "
                         "transparency. They act well because their actions are "
                         "seen by the community they belong to. Community norms "
                         "and shared values outline the code of conduct.",
        },
        "reference_overrides": {
            1: {
                "Corporation": "Employees are kept in line by means of "
                               "organizational culture, which relies on "
                               "internal values, group socialization, and 'the "
                               "way things are done here.'",
            },
            2: {
                "State": "It is most sensitive to officials and political "
                         "actors, such as regulators, ministries, and "
                         "legislators.",
                "Profession": "It is most sensitive to eminent scientists, high "
                              "ranking professionals and the judgment of the "
                              "research and professional communities.",
                "Market": "It is most sensitive to business analysts, "
                          "investors, and the business news world. As these are "
                          "the drivers of market success.",
                "Corporation": "It is most sensitive to its own leadership and "
                               "internal hierarchy.",
                "Family": "The head of the family is the most important when "
                          "assessing the conduct of members of the "
                          "organisation, the opinions of other family members "
                          "are also important.",
                "Religion": "It is most sensitive to the judgement of God, as "
                            "voiced through religious leaders, prophets, and "
                            "sacred texts.",
                "Community": "Its primary sensitivity lies in the judgment of "
                             "the community, specifically its members, research "
                             "community, and the broader movement.",
            },
            3: {
                "State": "Discipline operates through backroom politics, with "
                         "lobbying, alliances, and negotiation with government "
                         "and interest-group actors.",
                "Profession": "The discipline of the company is policed by high "
                              "ranking professionals and research scientists "
                              "who have a strong understanding of artificial "
                              "intelligence as a subject.",
                "Market": "Scrutiny from market analysts and the market "
                          "maintains discipline. Behaviour is kept in check by "
                          "press coverage of outcomes, the response of "
                          "investors, and standing in industry rankings.",
                "Corporation": "The firm itself delegates discipline, through "
                               "outside cultural norms and industry practices "
                               "that are enforced internally.",
                "Family": "Traditional family values, are informally enforced, "
                          "along with state guidelines about how the family "
                          "should be run.",
                "Religion": "Discipline operates through religious authority. "
                            "Divergence is condemned as sin or heresy against "
                            "the teachings of God.",
                "Community": "Actions are visible to many broader communities "
                             "and also to outside actors, they all place "
                             "pressure on the organisation.",
            },
        },
    },
    "Economic System": {
        "questions": [
            "How is {org} funded and how does it sustain its operations?",
            "How does {org} explain the relationship between its mission and "
            "its commercial activities — why does it say it makes money the way "
            "it does?",
            "Who captures the economic value {org} creates, and how are "
            "ownership, profit, and control distributed?",
        ],
        "reference_answers": {
            "State": "Welfare capitalism: funding and structure are oriented to "
                     "public benefit — capped returns, nonprofit oversight, "
                     "public-interest missions, or partnership with the state.",
            "Profession": "Personal capitalism: the enterprise is sustained by "
                          "the expertise and reputation of its practitioners, "
                          "who are its central economic asset.",
            "Market": "Market capitalism: revenue comes from competitive sale "
                      "of products and services, disciplined by price, demand, "
                      "and competition.",
            "Corporation": "Managerial capitalism: a professionally managed "
                           "firm whose hierarchy allocates capital, pursues "
                           "scale, and reinvests for growth.",
            "Family": "Family capitalism: ownership and control are "
                      "concentrated in a founder family or personal inner "
                      "circle.",
            "Religion": "Vocational (occidental) capitalism: economic activity "
                        "is treated as the dutiful expression of a calling "
                        "rather than a means to profit.",
            "Community": "Cooperative capitalism: value is shared among members "
                         "— open assets, commons-based production, and "
                         "cooperative or distributed ownership.",
        },
        "reference_overrides": {
            1: {
                "State": "The organisation is funded using public funds and "
                         "subsidies. If this is not directly the case, it is "
                         "still true that the organisation exists as part of a "
                         "larger economic system and it is benefitting from the "
                         "involvement of the state in keeping that system "
                         "intact.",
                "Profession": "The organisation is funded from private "
                              "investors and using the revenue that it "
                              "generates from it's operations. This is because "
                              "of the valuable product that has been created by "
                              "the professionals working at these companies.",
                "Market": "The organisation is funded through private "
                          "investment, if the market deems that the company is "
                          "valuable then investors will be attracted to it.",
                "Corporation": "The company is funded by providing its "
                               "artificial intelligence services to those who "
                               "are willing to pay.",
                "Family": "The organisation is formed by capital generated by "
                          "the family as a whole.",
                "Religion": "The organisation is funded by many different true "
                            "believers who wish to serve god.",
                "Community": "The organisation is funded by the artificial "
                             "intelligence community who has strong incentive "
                             "and also financial backing.",
            },
            2: {
                "State": "The mission of the organisation is to serve the "
                         "government, and by extension the people. The "
                         "commercial activities that it pursues are a means to "
                         "provide the most value to the state and public as "
                         "possible while also funding the continued development "
                         "of the technology.",
                "Profession": "The commercial activities are used to fund the "
                              "mission, it is clear that the hard working "
                              "professionals at these companies are eager to "
                              "develop artificial general intelligence, and the "
                              "commercial activities fuel that goal.",
                "Market": "The primary goal of the operation of the company is "
                          "to serve a market need and turn a profit. Creating "
                          "artificial intelligence models is a uniquely "
                          "lucrative opportunity so the company has decided to "
                          "pursue this.",
                "Corporation": "The company has an internal mission and its "
                               "commercial activities are in service to its "
                               "internal mission. The internal code of conduct "
                               "dictates how this mission may be achieved.",
                "Religion": "The commercial activities are only a means to an "
                            "end, and are wholly inservice to the mission of "
                            "serving god.",
            },
            3: {
                "State": "The economic value is captured by the government and "
                         "the people who live under the rule of the government. "
                         "The ownership should be given to the government, who "
                         "would use it for the betterment of the nation, "
                         "control should be in the hands of top government "
                         "officials.",
                "Profession": "The ownership belongs to the professionals who "
                              "have built these companies from the ground up, "
                              "by distributing shares to its workforce the "
                              "ownership, profit, and control is given to those "
                              "who have built these models.",
                "Market": "The investors and shareholder would be the "
                          "beneficiaries of the profit motive of the company, "
                          "furthermore market sentiment is the primary driver "
                          "of the decisions of the company, as this is how it "
                          "will generate a profit.",
                "Corporation": "The economic value should be captured by the "
                               "corporation as a way to increase its own size "
                               "and value. The shareholders and executives "
                               "should also be given a part of the value "
                               "generated.",
                "Religion": "All economic value should be generated for the "
                            "betterment of the mission and for religious "
                            "leaders and true believers.",
            },
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

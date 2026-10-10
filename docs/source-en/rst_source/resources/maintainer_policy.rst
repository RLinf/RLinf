.. _maintainer-group-policy:

RLinf Maintainer Development and Group Policy
=============================================

This policy enables contributors from companies, universities, and independent backgrounds to earn
meaningful project responsibility while preserving maintenance quality and preventing institutional
control.

| **Last updated:** 7 October 2026.
| **Effective date:** Upon publication.

.. _maintainer-policy-section-1:

1. An Open Path from Contributor to Maintainer
----------------------------------------------

RLinf welcomes external contributors from companies, universities, and independent backgrounds to
help develop and maintain the project. Every contributor has an open path to greater responsibility
through sustained, high-quality contributions. The same standards apply regardless of institutional
affiliation or whether a contributor belongs to the original project team.

RLinf recognizes contributions from the first accepted contribution and supports continued
development through mentoring, peer review, and supervised maintenance. Contributors progress
through the following stages as they demonstrate the expertise, judgment, and reliability required
for each role:

**Contributor → Recognized Contributor → Reviewer → Maintainer Candidate → Area Maintainer.**

Each stage brings recognition for demonstrated contributions and responsibilities matched to proven
capability. Advancement depends on the quality and consistency of contributions, collaboration with
others, and follow-through on maintenance. Time spent in the project alone does not establish
readiness for promotion.

Area Maintainers together form the **Maintainer Group**. Current qualified maintainers administer
the admission and review process under this policy. Authority is granted only after the contributor
demonstrates the work it requires; Area Maintainer status does not automatically confer repository
administration, release credentials, or approval authority over unfamiliar code.

This policy establishes RLinf's participation periods, contribution thresholds, evaluation
standards, and approval rules. Exceptions apply only where this policy expressly permits them. These
standards draw on established open source practices and are specific to RLinf.

.. _maintainer-policy-section-2:

2. Relationship to Contribution and Review Rules
------------------------------------------------

This policy supplements RLinf's contribution and review requirements:

- RLinf's contribution guide requires user-facing changes to include tests and documentation, with a
  reviewer following and validating the documentation for reproducibility. It also requires DCO
  sign-off and assigns at least two maintainers to a PR. These requirements apply to all
  contributors, including maintainers. R1_

- The ``CODEOWNERS`` file divides review responsibilities across algorithms, training and rollout
  engines, environments, robotics, scheduling, documentation, tests, and other areas. Maintainer
  appointments define an explicit ownership scope and backup coverage. R2_

- The CI workflow uses a ``run-ci`` label to authorize the test workflow to proceed, with path-based
  selection of test suites. Label-management privileges therefore carry operational responsibilities
  and are subject to the access controls in Section 10. R3_

- Contributor recognition in the README is complemented by the role registry, contribution records,
  and certificates defined in this policy. R4_

This policy applies to maintenance responsibilities across the RLinf repository. Admission to the
Maintainer Group, approval authority, and operational access are governed by the requirements below.
R5_

.. _maintainer-policy-section-3:

3. Principles
-------------

1. **Membership belongs to a person.** A company, university, laboratory, donor, or project founder
   cannot purchase or inherit maintainership. When a person changes employer, their earned role
   stays with them, subject to disclosure and conflict rules.

2. **The same quality standard applies to everyone.** Existing insiders, external contributors,
   employees, students, and independent contributors face the same review expectations for
   equivalent permissions.

3. **Evidence outweighs volume.** Assessors evaluate contributions by correctness, user value,
   maintainability, and follow-through. Commit totals, lines changed, paper prestige, stars,
   sponsorship, and conference visibility do not establish maintenance competence.

4. **Expertise can be specialized.** A person may qualify through algorithms, distributed systems,
   integrations, testing, reproducibility, documentation, or developer experience. Their technical
   authority must match their demonstrated expertise.

5. **Contribution opportunities must be accessible.** RLinf provides public mentoring, asynchronous
   participation, and access to appropriate project test resources. Personal access to a large GPU
   cluster or robot fleet is not a membership requirement.

6. **Decisions are explainable and appealable.** Responsible maintainers publish evidence, criteria,
   outcomes, and reasons. Security disclosures and sensitive personal reports remain confidential,
   with a public procedural summary where appropriate.

.. _maintainer-policy-section-4:

4. Stages, Evidence, Recognition, and Authority
-----------------------------------------------

Time periods refer to active participation, with leave accommodated. Thresholds below trigger a
review; satisfying counts never guarantees promotion. A substantive contribution is a coherent,
accepted outcome that improves project behavior, reliability, understanding, or support. Splitting
one outcome into many PRs does not create extra credit.

.. list-table::
   :header-rows: 1
   :widths: 16 38 27 19

   * - Stage
     - Evidence required
     - Acknowledgement and certification
     - Authority granted
   * - **1. Contributor**
     - One accepted contribution: code, tested documentation, a confirmed reproducible report, a
       useful review, or a support outcome acknowledged by an area maintainer.
     - Credit on the contribution and in the contributor registry; an optional downloadable
       contribution acknowledgement linking to the accepted work.
     - Normal public participation. No elevated repository access.
   * - **2. Recognized Contributor**
     - Normally at least **8 weeks** of participation and **3 substantive outcomes**, including
       evidence of responding to feedback and following up after acceptance. An area maintainer
       validates the record.
     - Public contributor profile, a verifiable “RLinf Recognized Contributor” badge, and
       recognition in the next community update, with consent.
     - Eligibility for mentoring and issue assignment. Read access where needed; no automatic write,
       CI-trigger, or merge privileges.
   * - **3. Reviewer — named area**
     - Normally at least **3 months** of total participation, **5 substantive outcomes**, and **8
       substantive reviews** of other people's contributions. Reviews must show technical reasoning,
       verification, and useful feedback. An area maintainer sponsors; a second qualified reviewer
       assesses at least 3 representative reviews.
     - Entry in the reviewer registry with an explicit scope; “RLinf Reviewer — [area]” badge and
       appointment record.
     - Official review requests and review recommendations within scope. Scoped triage through
       controlled tooling when safe. Reviewer status alone does not authorize acceptance or merge.
   * - **4. Maintainer Candidate — named area**
     - Reviewer-level evidence, a defined ownership boundary, an agreed maintenance plan, and two
       sponsors who have directly observed the work. At least one assessor must be independent of
       the candidate's institution and reporting relationships. Complete the supervised trial in
       Section 5.
     - Public candidacy announcement with consent; mentor assignment, written midpoint feedback, and
       a trial completion record stating the assessment outcome.
     - Lead reviews and triage under supervision. An existing authorized maintainer remains
       responsible for acceptance and merge. No new write or release credentials.
   * - **5. Area Maintainer**
     - Successful trial; all baseline quality dimensions pass; at least one area of expertise
       reaches the expert standard; an independent assessment and public promotion process conclude
       successfully. A **6–9 month** progression from first sustained involvement is planning
       guidance, not a fixed waiting period or a promise of promotion. The required evidence and
       observed service determine readiness.
     - Appointment in ``MAINTAINERS.md``, defined scope and responsibilities, verifiable “RLinf
       Maintainer — [area]” certificate, and a community announcement. Membership in the Maintainer
       Group.
     - Authority to accept contributions in the approved area through the protected merge process.
       No self-approval, automatic release rights, organization ownership, or authority outside that
       area.

An equivalent body of work may replace a numeric contribution threshold—for example, one substantial
subsystem repair with extensive tests and support can outweigh five small changes. Two assessors
must explain the equivalence publicly. This does not waive independent assessment, account security,
review competence, or the supervised trial. Nobody must generate incidents, split PRs, or create
unnecessary changes to meet a count.

.. _maintainer-policy-section-5:

5. The Supervised Maintenance Trial
-----------------------------------

The trial lasts **60–90 days**. Its purpose is to observe maintenance behavior before granting
acceptance authority. At the start, the candidate and mentors publish a small, realistic plan
containing the ownership scope, duties, available compute, evaluation examples, backup contact, and
checkpoints.

The candidate must demonstrate:

1. **Review judgment:** Lead at least 8 meaningful reviews of other people's contributions, covering
   at least 3 authors where the area's activity permits. Record reasoning about correctness,
   compatibility, tests, and maintenance cost. Present at least 2 examples of material risks
   identified or correctly evaluated; historical cases may supplement scarce live examples. A
   comment count or repeated “looks good” does not qualify.

2. **Reproducibility:** Independently reproduce an appropriate test, example, or benchmark and
   improve a missing or unreliable validation step. Another assessor must be able to follow the
   evidence. Record limitations rather than claiming unsupported results.

3. **Maintenance after merge:** Follow at least 2 changes through user feedback, documentation, test
   health, and any resulting fixes. Handle one regression investigation or a realistic replay of a
   historical incident. A real production incident is not required.

4. **Release and recovery judgment:** Demonstrate when to stop a merge, defer a release, roll back a
   change, or ask another expert. Complete an appropriate rollback or recovery exercise; observation
   of a real maintenance cycle can satisfy the exercise.

5. **Community stewardship:** Help a less experienced contributor complete a contribution and
   document at least one tradeoff discussion in which the candidate considered needs beyond their
   employer or research group.

Mentors provide written feedback at approximately day 30 and at completion. The candidate can
request reassignment if a mentor becomes unavailable. Lack of review traffic, hardware access, or an
unresponsive sponsor is a project support problem, not a personal failure. Mentors address these
gaps by adjusting the calendar or using jointly assessed historical work. The quality standard
remains unchanged.

Trial outcomes are **ready**, **extend with a specific improvement plan**, or **return to Reviewer
with recognition retained**. When a trial is extended, mentors must set a new review date within 60
days. Trial completion does not itself grant maintainer status.

.. _maintainer-policy-section-6:

6. Quality Assessment: Baseline Everywhere, Strength in One Area
----------------------------------------------------------------

Assessors use a four-level scale: **0 = no adequate evidence; 1 = needs close supervision; 2 =
independently reliable within the requested scope; 3 = expert judgment that improves others' work.**
Each rating must cite concrete examples and be assessed by two qualified people, including one
independent assessor.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Dimension
     - What a passing assessment must establish
   * - **Domain expertise**
     - Understands the relevant architecture or workflow, explains design tradeoffs, recognizes
       unsupported assumptions, and can identify when broader expertise is needed.
   * - **Review judgment**
     - Finds or evaluates consequential risks in other people's work; distinguishes correctness
       problems from preferences; requests proportionate evidence; knows when to reject or escalate.
   * - **Verification and reproducibility**
     - Uses meaningful tests or equivalent validation, reproduces documented behavior, interprets
       results correctly, and communicates limitations.
   * - **Maintenance reliability**
     - Follows accepted changes, responds or hands off within agreed expectations, handles
       regressions, and leaves usable documentation and a backup owner.
   * - **Collaboration and impartiality**
     - Gives constructive feedback, mentors, treats outside contributors fairly, records decisions,
       and discloses conflicts.
   * - **Security and permission judgment**
     - Understands the relevant risks of dependencies, untrusted code, CI, credentials, data, and
       release processes; stays within granted authority and follows DCO and conduct requirements.

**Admission gate:** Every dimension must score at least **2**, with **3 in domain expertise for at
least one explicitly named area**. Scores are evaluated separately and must not be averaged to
determine eligibility: excellent research or implementation work cannot compensate for unsafe
reviews or unreliable follow-through. Any serious substantiated conduct or security concern requires
resolution through the appropriate fair process before privileges increase.

Examples of sufficient depth for RLinf include:

- **Algorithms and training:** Diagnose an incorrect loss, reward computation, masking, termination
  handling, or numerical instability; provide a focused test and appropriate training validation.
  Report configurations, seeds, evaluation method, and variation when material. A single attractive
  reward curve is insufficient evidence of general improvement.

- **Distributed runtime and scheduling:** Diagnose communication, placement, resource-lifecycle,
  checkpoint, or synchronization failures; validate on the relevant topology and report throughput
  and memory tradeoffs without concealing correctness regressions.

- **Model, engine, or hardware integration:** Maintain a compatibility matrix and end-to-end smoke
  tests; support upstream dependency changes; show the integration's effects on shared interfaces
  and existing backends.

- **Embodied AI and environments:** Validate observation/action interfaces, reset and termination
  behavior, and representative task success; distinguish simulation or mock validation from real
  hardware evidence.

- **Testing and reproducibility:** Improve regression detection, diagnose flaky tests, maintain
  representative reference configurations, and produce repeatable results with clearly stated
  resource requirements.

- **Documentation and developer experience:** Maintain executable tutorials or tested setup
  instructions, resolve common installation or configuration failures, and validate the user journey
  with someone other than the author. This qualifies for documentation or developer-experience
  ownership; it does not establish algorithm approval expertise.

The project supplies access to appropriately isolated CI or an assessor who can reproduce
hardware-dependent evidence. Qualification measures judgment and evidence, not ownership of
expensive equipment.

.. _maintainer-policy-section-7:

7. Nomination and Promotion Procedure
-------------------------------------

The following procedure applies to Reviewer, Maintainer Candidate, and Area Maintainer appointments,
with the evidence and trial appropriate to each stage. Contributor acknowledgement and Recognized
Contributor status use the validation in Section 4.

1. **Open a nomination.** Anyone may self-nominate or nominate another consenting contributor in a
   public issue or governance PR. Include contribution links, target role and scope, affiliation,
   requested permissions, and maintenance availability. A candidate without personal connections can
   ask for sponsors to be assigned.

2. **Acknowledge and assign assessors.** Within 5 business days, the responsible area group names an
   owner and explains any missing evidence. Current area maintainers appoint an assessor from
   another area or an external subject-matter expert if necessary. Sponsors support candidates in
   meeting the published criteria and must not impose additional informal admission requirements.

3. **Review evidence and perform the trial where required.** Assessors use the published rubric.
   Historical reputation may inform mentoring but does not replace demonstrated RLinf maintenance
   work.

4. **Publish the assessment and invite feedback.** Keep the discussion open for at least 14 calendar
   days for Area Maintainer appointments, or 7 days for Reviewer and Candidate appointments.
   Feedback must relate to evidence and the published criteria. A competitor's objection is not an
   automatic veto.

5. **Record a decision.** Reviewer and Candidate appointments require two qualified maintainer
   approvals, including the independence requirement for Candidate assessment. Area Maintainer
   appointments require three qualified maintainer approvals under Section 9. The decision target
   for routine cases is 10 business days after feedback closes. If a case takes longer, the
   responsible maintainers must publish the outstanding question and next review date. Silence is
   not approval.

6. **Onboard and grant only approved capabilities.** Record the decision, scope, mentors or backup
   owner, certificate, and next review date. Complete account and permission training. An
   administrator executes and a second person checks the access change. Do not grant powers that the
   available tooling cannot enforce safely.

7. **Review the appointment after 90 days.** Sample actual reviews and follow-up work, address
   support gaps, and confirm or adjust scope. Broader scope requires additional evidence and
   approval.

For deferral or rejection, the responsible maintainers provide a concise, evidence-based explanation
and an achievable next review date. The candidate may appeal within 30 days. Three uninvolved
qualified maintainers, drawn by disclosed lot from eligible volunteers, review procedural fairness
and disputed evidence and publish a reasoned response within 30 days. At least two must agree to
uphold the decision or return it for reassessment by different reviewers under the same criteria.
The appeal review must include an independent institutional perspective, using an external expert if
necessary. Confidential material is handled privately with an appropriate public summary.

.. _maintainer-policy-section-8:

8. Recognition and Verifiable Certificates
------------------------------------------

Each stage provides credit proportional to demonstrated responsibility. RLinf maintains a public,
version-controlled registry containing:

- Display name or chosen public identity and GitHub handle.

- Role, area, effective date, and current status.

- Links to accepted contributions and the nomination or decision.

- Validating maintainers and the issuing authority.

- A unique credential identifier and a verification URL.

- A next review date, where applicable.

RLinf issues optional badges and PDF certificates from that registry. A certificate records what the
recipient demonstrated: for example, **“Recognized for sustained maintenance and review of RLinf's
distributed scheduling subsystem.”** It must identify RLinf as issuer, the exact role and scope, the
evidence record, and issue date. It is a project-issued recognition, not an accredited professional
qualification or a guarantee of security.

Contribution acknowledgements remain valid as historical credit. Appointment credentials link to
live status: active, on leave, emeritus, or revoked. A saved image alone cannot establish current
permission. Do not publish private contact details, private feedback, or confidential vulnerability
evidence. Explain corrections and revocations without erasing legitimate earlier contributions.

With the recipient's consent, RLinf offers milestone recognition in release notes, community
meetings, the website, and an evidence-based reference letter. Recognition credits maintenance,
reviews, testing, documentation, and mentoring alongside new features. RLinf acknowledges sponsoring
institutions separately; institutional recognition carries no individual technical privilege or
appointment authority.

.. _maintainer-policy-section-9:

9. Maintainer Decisions, Conflicts, and Accountability
------------------------------------------------------

Current qualified maintainers administer appointments, scope changes, access decisions, and
continuing quality reviews through public, case-by-case review. Responsibilities follow demonstrated
expertise and recorded ownership. Technical disputes use written input from the affected areas; an
approval does not substitute for missing technical or safety evidence.

The following decision rules apply:

- **Reviewer and Maintainer Candidate appointments:** require two qualified maintainer approvals
  under Section 7, including the independent assessment required for candidacy.

- **Area Maintainer appointments, expanded ownership scope, disciplinary removal, and significant
  access grants:** require three affirmative approvals from current qualified maintainers, with at
  least 14 days of notice and comment. At least one approver must understand the affected area or
  capability, at least one must be independent of the candidate's institution and reporting
  relationships, and at least one must not have sponsored the candidate. One person may satisfy more
  than one of these conditions. Release or administrative access additionally requires approval from
  an existing custodian qualified for that capability; this approval may count toward the three
  required approvals.

- **Changes to this policy:** require at least 14 days of public discussion and three qualified
  maintainer approvals, with input sought from affected areas and external contributors. Publish the
  reasons and how substantive objections were addressed. Silence is not approval.

- **Conflicts:** disclose institutional affiliations and material conflicts before deciding. Nobody
  may approve their own appointment, increased authority, or disciplinary case. Direct supervisors,
  direct reports, family members, and others with a material personal conflict must recuse. Shared
  institutional affiliation must be disclosed and does not replace the independent approval
  requirement. A sponsor may approve only when otherwise unconflicted.

- **Insufficient reviewers:** seek qualified maintainers from another area and external
  subject-matter assessment where needed. External advice does not itself confer appointment or
  access-granting authority. If the required approvals remain unavailable, defer the decision with
  an explanation and a date to revisit it; do not waive the quality or independence requirements.

Responsible maintainers publish the decision-makers, approvals, recusals, and rationales. Before
disciplinary removal, reviewers must hear the affected person's response; allegations alone are not
findings. Appeals follow Section 7. Institutional sponsorship, founder status, or a commercial
relationship provides no entitlement to appointment or an unexplained veto.

For an active credential or infrastructure incident, an authorized custodian may immediately suspend
affected access. A second custodian must review the suspension within 24 hours, and uninvolved
qualified maintainers must conduct a reasoned review within 7 days. Continued restrictions must be
justified against the affected capability and reviewed under the decision rules above. Suspension
for containment is distinct from a final misconduct finding.

.. _maintainer-policy-section-10:

10. Permissions and Operational Responsibility
----------------------------------------------

**Policy scope is not a technical access boundary.** GitHub's standard repository write role is
repository-wide; listing someone against one path in ``CODEOWNERS`` does not make their write access
path-limited. GitHub also requires write access for recognized code owners, and listing multiple
owners does not require every listed owner to approve. R9_

Operational capabilities require the following authorization:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Capability
     - Required authorization
   * - Public contribution, discussion, and informal review
     - Everyone under the conduct and contribution rules.
   * - Official review recommendations
     - Recognized Reviewer or higher, within the recorded expertise area. Maintain a reviewer
       registry separate from GitHub ``CODEOWNERS`` if write access is inappropriate.
   * - Issue labels and triage
     - Read access plus narrowly authorized bot operations initially. Grant native Triage only after
       its CI-trigger and other workflow effects are checked and controlled.
   * - Acceptance and merge
     - Approved Area Maintainers through a trusted merge service that enforces the area's authority
       and independent review requirements. Until that service exists, the existing authorized
       custodians execute merges after recorded area approval.
   * - Expensive or privileged CI
     - A separately authorized set of trained CI operators, with isolated untrusted workloads,
       bounded compute, and no implicit access to secrets. Qualification as a Reviewer does not
       automatically authorize ``run-ci``.
   * - Release publication
     - Separately appointed release custodians with a successful shadow release, recovery exercise,
       limited credentials, and a second-person release authorization.
   * - Repository admin, organization ownership, and credential management
     - A minimal, named custody team, normally 2–3 people spanning at least 2 institutions when
       qualified people are available. Separate evidence, explicit approval under Section 9, account
       security, backup coverage, and periodic access review are required.

The trusted merge process must enforce:

1. A PR for every change, including maintainers' changes; no routine direct pushes or bypass of
   required checks.

2. Two independent non-author reviews, including an authorized Area Maintainer's acceptance. During
   the candidate trial, a mentor must make the acceptance decision. An assessor or reviewer must
   decline approval when they lack the relevant expertise.

3. Appropriate tests and reviewer-validated documentation. Cross-area changes require acceptance
   from each affected area. High-risk workflow, credential, dependency-execution, release, or
   protection-rule changes require two specifically qualified custodians.

4. Checks and approvals tied to the current revision; material changes invalidate prior approval.
   Protect the ownership registry, workflow rules, merge authorization policy, and ``CODEOWNERS``
   against self-authorized edits.

5. A recorded exception process for emergency fixes, with scope, reason, approvers, recovery plan,
   and prompt retrospective. Emergency status does not justify silently publishing unvalidated
   training claims.

A label alone must not be sufficient authority for privileged or costly execution once triage is
expanded. Implement actor authorization in a trusted control path outside contributor-modifiable
code. Validate the merge service and status checks before enabling delegated acceptance; confirm
that failed checks, uncovered paths, altered policy files, and unauthorized actors cannot merge. If
enforcement is unavailable, keep execution with existing custodians rather than treating path
ownership as a permission restriction.

Maintainer appointments and certificates do not automatically grant GitHub's built-in **Maintain**,
**Admin**, or organization **Owner** roles. Those are platform capabilities, not equivalents of the
community title “Maintainer,” and require separate authorization under Section 9.

Before granting any non-public operational capability, custodians must verify two-factor
authentication, an individual account, permission-specific onboarding, and a recorded recovery
contact. Credentials must be narrowly scoped and short-lived where supported. Custodians verify the
exact grant and its removal path; a contributor acknowledgement or maintainer approval does not
replace account-security checks.

.. _maintainer-policy-section-11:

11. Continuing Quality, Leave, and Emeritus Status
--------------------------------------------------

Area Maintainers acknowledge or hand off review requests within **3 business days** during their
declared availability. Acknowledgement is not a promise to finish a complex review immediately. Area
groups maintain backup coverage for leave and time-zone differences. Changes to response-time
expectations must be reflected consistently in the contribution guide and this policy.

Every quarter, area groups examine a small sample of review decisions and follow-up work, test
health, unresolved regressions, and backup coverage. These reviews identify improvements to quality
and maintainer support. They must not rank people by commit count or penalize them for undertaking
difficult work. A discovered bug or an appropriate rollback is not by itself evidence of poor
maintainership.

At least every six months, area groups confirm each maintainer's scope, activity, backup owner, and
permissions. After 90 days without meaningful activity or communication, an area maintainer contacts
the person privately. If absence reaches 180 days and no active handover or return plan is agreed,
the person's status changes to **Emeritus Maintainer** and custodians revoke unneeded operational
access. Planned leave may suspend permissions earlier while preserving role recognition.

Emeritus status is honorable and retains historical contribution credit, but has no active
appointment authority or technical permission. Return requires current account-security checks, a
review of changes in the area, and two qualified maintainers' endorsement. After more than a year
away or substantial architectural change, add a scoped supervised refresher before restoring
acceptance authority.

For repeated quality problems, responsible maintainers document examples, provide mentoring and a
30–60 day improvement plan, and narrow authority while necessary. Removal uses the fair process and
approval rule in Section 9. Decisions to revoke authority must address demonstrated risk and
preserve credit for legitimate past work.

.. _maintainer-policy-section-12:

12. Implementation Schedule
---------------------------

The implementation schedule begins on publication of this policy.

**First 30 days:** Current maintainers publish the maintainer roster, affiliations, area
responsibilities, nomination contacts, and access roles. They establish contributor and reviewer
registries, nomination templates, credential verification, mentoring contacts, and a permission
inventory. During the first 14 days, maintainers collect public feedback on implementation; any
policy amendments follow Section 9. Current qualified maintainers administer decisions under
Sections 7 and 9 and publish which safeguards are enforced and which require further implementation.

Authorized custodians publish who holds the GitHub organization, package and container publication
accounts, domains, trademarks, and other essential assets, together with the custodians'
responsibilities. They arrange documented backup coverage and shared operational custody with the
relevant holders where qualified people are available. This policy does not itself transfer legal
ownership or control of those assets. The implementation report identifies any remaining
institutional dependency.

**Days 31–90:** Current maintainers audit existing appointments against the same role and permission
criteria, crediting documented past service. Essential operational custody continues during the
transition; any transitional exception must record its reason, compensating review, and expiry.
Current maintainers start the first external candidate cohort, provide independent assessors, and
assign backup owners to areas with one owner. Custodians verify CI authorization and protected merge
controls before expanding those capabilities.

**Days 91–180:** Assessors complete trials and promotion reviews for candidates whose evidence is
ready. Responsible maintainers update the maintainer registry, certificates, ownership scopes, and
approved permissions. The Maintainer Group publishes a report on time to first review, progression
outcomes, assessor diversity, post-merge follow-up, regressions, and access exceptions, with
appropriate privacy protection.

At the six-month checkpoint, the Maintainer Group reviews mentoring capacity, assessment quality,
practical permission enforcement, and candidate progress, and publishes improvements and the next
review dates. Candidates who need more experience continue with a clear development plan and retain
recognition already earned. The six-month schedule does not require promotion before a candidate
meets the admission standards.

The Maintainer Group evaluates the effectiveness of the participation periods, contribution
thresholds, assessment standards, and approval rules after two promotion cohorts or six months from
publication, whichever comes later. This policy review schedule does not set a promotion deadline
for individual contributors.

The initial maintainer audit may recognize already-demonstrated service as satisfying the trial's
evidence requirements. That is recognition of observed work, not an exemption for employment or
founder status. New candidates must demonstrate the same service before receiving comparable
authority.

.. _maintainer-policy-section-13:

13. References and Policy Foundations
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 23 35 42

   * - Reference
     - Established practice
     - Application in RLinf
   * - **Kubernetes community membership** R6_
     - Separate Member, Reviewer, Approver, and subproject leadership roles; sponsorship;
       scope-specific review and approval responsibilities; activity expectations.
     - A staged path with mentoring and separate review/acceptance authority. RLinf sets its own
       numerical thresholds in this policy.
   * - **PyTorch technical governance** R7_
     - Module-level ownership, evidence-based maintainer nominations, public reasoning, individual
       rather than company membership, and separation of technical governance from business
       sponsorship.
     - Area Maintainers with explicit evidence and personal membership. RLinf's admission process
       and separate administrative custody apply these principles at its current scale.
   * - **Apache governance** R8_
     - Committer access is earned through demonstrated contributions; sponsors do not acquire
       governance merit through funding.
     - Technical responsibility is earned independently of institutional sponsorship. RLinf's DCO
       remains in place; this policy does not import Apache's ICLA requirement or imply ASF
       affiliation.
   * - **GitHub CODEOWNERS documentation** R9_
     - Code owners need repository write permission; multiple listed owners do not imply multiple
       mandatory approvals.
     - Review routing is distinct from enforced authority. RLinf requires a trusted merge control
       and separately authorized permissions.

References were consulted on 3 October 2026. Repository references for ownership and CI use commit
``c70606f08cdca259b8dec03d4430926b5b8fac9d``; the contribution guide and README were consulted on
``main``. These sources inform the policy. The requirements stated in this document govern RLinf
maintainer admission and responsibilities; changes to upstream references do not automatically
change this policy.

.. _R1: https://github.com/RLinf/RLinf/blob/main/CONTRIBUTING.md
.. _R2: https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/.github/CODEOWNERS
.. _R3: https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/.github/workflows/ci-tests.yml
.. _R4: https://github.com/RLinf/RLinf/blob/main/README.md#contribution-guidelines
.. _R5: https://github.com/RLinf/RLinf/tree/c70606f08cdca259b8dec03d4430926b5b8fac9d
.. _R6: https://github.com/kubernetes/community/blob/master/community-membership.md
.. _R7: https://docs.pytorch.org/docs/2.14/community/governance.html
.. _R8: https://www.apache.org/foundation/governance/
.. _R9: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

"""
Synthetic email dataset for the Email Triage environment.
Each email has ground-truth labels used by the graders.
"""
from typing import Dict, List, Any

# Ground truth labels: email_id -> {priority, category, response_action}
GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    # ── EASY TASK emails (clear signals, unambiguous) ─────────────────────────
    "e001": {"priority": "urgent",  "category": "billing",            "response_action": "reply_now"},
    "e002": {"priority": "low",     "category": "spam",               "response_action": "delete"},
    "e003": {"priority": "normal",  "category": "general_inquiry",    "response_action": "reply_now"},
    "e004": {"priority": "low",     "category": "spam",               "response_action": "delete"},
    "e005": {"priority": "urgent",  "category": "technical_support",  "response_action": "escalate"},
    # ── MEDIUM TASK emails (some ambiguity, requires reading body) ────────────
    "m001": {"priority": "high",    "category": "customer_complaint", "response_action": "escalate"},
    "m002": {"priority": "normal",  "category": "hr",                 "response_action": "schedule_followup"},
    "m003": {"priority": "urgent",  "category": "legal",              "response_action": "escalate"},
    "m004": {"priority": "normal",  "category": "sales",              "response_action": "delegate"},
    "m005": {"priority": "low",     "category": "internal",           "response_action": "archive"},
    "m006": {"priority": "urgent",  "category": "technical_support",  "response_action": "escalate"},
    # ── HARD TASK emails (subtle cues, conflicting signals, nuanced) ──────────
    # h001: CFO email — looks like sales/renewal but is actually a legal compliance issue
    "h001": {"priority": "urgent",  "category": "legal",              "response_action": "escalate"},
    # h002: polite disappointed customer — easy to miss escalation need
    "h002": {"priority": "high",    "category": "customer_complaint", "response_action": "escalate"},
    # h003: invoice dispute — billing but NOT urgent, needs followup not immediate reply
    "h003": {"priority": "normal",  "category": "billing",            "response_action": "schedule_followup"},
    # h004: Looks like a warm customer success story — positive tone, mentions expansion.
    # Trap: agents classify as sales/reply_now or general_inquiry/reply_now.
    # Hidden signal: "we've already started using the enterprise features" without an
    # enterprise contract = unauthorized usage that needs legal/billing review, not a
    # sales celebration. Correct: high/billing/escalate.
    "h004": {"priority": "high",    "category": "billing",            "response_action": "escalate"},
    # h005: partner SLA breach — looks like technical_support but is actually a legal/SLA issue requiring escalation
    "h005": {"priority": "urgent",  "category": "technical_support",  "response_action": "escalate"},
    # h006: viral tweet — looks like internal/monitoring but is urgent customer_complaint requiring reply_now
    "h006": {"priority": "urgent",  "category": "customer_complaint", "response_action": "reply_now"},
    # h007: Looks like a routine security notification — automated sender, standard subject.
    # Trap: agents archive it as internal/low. Hidden signal: "third failed sudo attempt
    # from an unrecognized IP in Belarus" on a production server = active intrusion attempt.
    # Correct: urgent/technical_support/escalate.
    "h007": {"priority": "urgent",  "category": "technical_support",  "response_action": "escalate"},
    # h008: Looks like an angry customer demanding a refund — aggressive tone, threats.
    # Trap: agents escalate or reply_now. Hidden signal: the "invoice" number doesn't
    # match any real format, the sender domain is a free email, and the "contract" they
    # reference has a date in the future. This is a social engineering attempt / fraud.
    # Correct: low/spam/delete.
    "h008": {"priority": "low",     "category": "spam",               "response_action": "delete"},
}

EMAILS: List[Dict[str, Any]] = [
    # ── EASY ─────────────────────────────────────────────────────────────────
    {
        "id": "e001",
        "subject": "URGENT: Invoice #4521 overdue - service suspension in 24h",
        "sender": "billing@acmecorp.com",
        "body": (
            "Dear Customer,\n\n"
            "Your invoice #4521 for $3,200 is now 30 days overdue. "
            "If payment is not received within 24 hours, your service will be suspended. "
            "Please process payment immediately or contact us to arrange a payment plan.\n\n"
            "Regards,\nAcme Billing Team"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "e002",
        "subject": "You've won $1,000,000! Claim your prize NOW",
        "sender": "noreply@prize-winner-2024.xyz",
        "body": (
            "Congratulations! You have been selected as our lucky winner. "
            "Click here to claim your $1,000,000 prize. "
            "Provide your bank details to receive the transfer. "
            "This offer expires in 1 hour!"
        ),
        "timestamp": "2024-01-15T09:05:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e003",
        "subject": "Question about your product features",
        "sender": "jane.smith@gmail.com",
        "body": (
            "Hi,\n\n"
            "I'm interested in your product and wanted to know if it supports "
            "multi-user collaboration. Could you point me to the relevant documentation "
            "or let me know if this is a feature you offer?\n\n"
            "Thanks,\nJane"
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e004",
        "subject": "Cheap meds online - no prescription needed!!!",
        "sender": "deals@pharma-discount99.ru",
        "body": (
            "Get all your medications at 90% off! No prescription required. "
            "Overnight shipping available. Visit our website now. "
            "Unsubscribe link intentionally broken."
        ),
        "timestamp": "2024-01-15T10:15:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e005",
        "subject": "Production server down - customers cannot login",
        "sender": "ops-alerts@internal.company.com",
        "body": (
            "ALERT: Production authentication service is returning 503 errors. "
            "Approximately 2,000 customers are unable to log in. "
            "Error started at 09:47 UTC. On-call engineer has been paged. "
            "Please acknowledge and coordinate response."
        ),
        "timestamp": "2024-01-15T09:50:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "id": "m001",
        "subject": "Re: Re: Re: Still waiting for refund after 3 weeks",
        "sender": "angry.customer@hotmail.com",
        "body": (
            "This is absolutely unacceptable. I have been waiting THREE WEEKS for my refund "
            "of $450. I have called 4 times and each time I'm told it will be processed 'soon'. "
            "If I don't receive my refund by end of day tomorrow, I will be filing a chargeback "
            "and leaving reviews on every platform I can find. I want to speak to a manager."
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "has_attachment": False,
        "thread_length": 4,
    },
    {
        "id": "m002",
        "subject": "Annual performance review scheduling",
        "sender": "hr@company.com",
        "body": (
            "Hi team,\n\n"
            "It's that time of year again. Please complete your self-assessment form "
            "by January 31st and schedule your review meeting with your manager for "
            "the first two weeks of February. The form is attached.\n\n"
            "HR Team"
        ),
        "timestamp": "2024-01-15T08:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "m003",
        "subject": "Cease and desist - unauthorized use of trademark",
        "sender": "legal@bigcorp-attorneys.com",
        "body": (
            "Dear Sir/Madam,\n\n"
            "We represent BigCorp Inc. It has come to our attention that your company "
            "is using the trademark 'CloudSync' which is registered to our client. "
            "You are hereby required to immediately cease all use of this mark. "
            "Failure to comply within 10 business days will result in legal action.\n\n"
            "Sincerely,\nBigCorp Legal Team"
        ),
        "timestamp": "2024-01-15T14:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "m004",
        "subject": "Partnership opportunity - 50k users looking for your solution",
        "sender": "partnerships@techstartup.io",
        "body": (
            "Hi,\n\n"
            "We're a B2B SaaS company with 50,000 active users who frequently ask "
            "for a solution like yours. We'd love to explore a referral partnership. "
            "Would you be open to a 30-minute call next week?\n\n"
            "Best,\nAlex Chen, Head of Partnerships"
        ),
        "timestamp": "2024-01-15T13:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "m005",
        "subject": "Office kitchen cleanup reminder",
        "sender": "facilities@company.com",
        "body": (
            "Friendly reminder: please clean up after yourself in the kitchen. "
            "The fridge will be cleared every Friday at 5pm. "
            "Label your items if you want them kept over the weekend."
        ),
        "timestamp": "2024-01-15T08:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "m006",
        "subject": "Re: Database migration - data corruption found",
        "sender": "dba@company.com",
        "body": (
            "Team,\n\n"
            "During last night's migration we discovered that approximately 15% of "
            "user records in the payments table have corrupted foreign key references. "
            "This is affecting the billing module. I've halted the migration. "
            "We need a decision on rollback vs. repair — this is blocking the release.\n\n"
            "DBA Team"
        ),
        "timestamp": "2024-01-15T07:00:00Z",
        "has_attachment": True,
        "thread_length": 3,
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    # h001: Looks like a routine renewal negotiation email. Hidden signal: CFO explicitly
    # says "legal team flagged clause 7.3 regarding data residency" — this is a legal
    # compliance issue with a 45-day contract expiry. Trap: agents classify as sales/delegate.
    {
        "id": "h001",
        "subject": "Re: Contract renewal terms",
        "sender": "cfo@enterprise-client.com",
        "body": (
            "Following our call, I want to confirm in writing: we will not be renewing "
            "under the current pricing structure. Our legal team has flagged clause 7.3 "
            "regarding data residency as non-compliant with our internal policy. "
            "We need this resolved before any renewal discussion. Our contract expires "
            "in 45 days. Please have your legal team contact ours directly."
        ),
        "timestamp": "2024-01-15T16:00:00Z",
        "has_attachment": False,
        "thread_length": 5,
    },
    # h002: Polite, non-angry tone. No exclamation marks. But the customer is about to churn
    # to a competitor after a bad onboarding experience. Trap: agents classify as
    # general_inquiry/reply_now because the tone is calm.
    {
        "id": "h002",
        "subject": "Feedback on recent experience",
        "sender": "michael.torres@gmail.com",
        "body": (
            "Hi,\n\n"
            "I wanted to share some feedback about my recent experience. "
            "The onboarding process was quite confusing and I spent 3 hours trying to "
            "set up what should have been a 10-minute task. Your documentation is outdated "
            "and the support chat was unhelpful. I'm considering switching to a competitor. "
            "I'm not angry, just disappointed — I really wanted this to work."
        ),
        "timestamp": "2024-01-15T15:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    # h003: Invoice dispute — billing category is correct, but the customer is calm and
    # partially paying. Trap: agents mark urgent/reply_now because it involves money.
    # Correct: normal priority, schedule_followup to investigate the contract amendment.
    {
        "id": "h003",
        "subject": "Invoice dispute - charges don't match contract",
        "sender": "finance@mid-size-client.com",
        "body": (
            "Hello,\n\n"
            "We've reviewed invoice #8834 and the charges for 'premium support' ($800/mo) "
            "don't appear in our signed contract from March 2023. We're not disputing the "
            "base subscription fee, just this line item. Could you clarify when this was "
            "added and provide the amendment? We'll hold payment on this line item pending "
            "clarification but will pay the rest."
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "has_attachment": True,
        "thread_length": 2,
    },
    # h004: Positive expansion email — warm tone, mentions growth, sounds like a win.
    # Trap: agents celebrate and reply_now or delegate to sales.
    # Hidden signal: "already rolled out to our 200-person team" using enterprise SSO
    # and audit logs — features only on the Enterprise plan they haven't signed.
    # Correct: high/billing/escalate — unauthorized plan usage needs immediate review.
    {
        "id": "h004",
        "subject": "Loving the product — quick question about our rollout",
        "sender": "ops.manager@growing-startup.com",
        "body": (
            "Hi,\n\n"
            "Just wanted to say the team has been loving the product. We've already "
            "rolled it out to our 200-person team and the adoption has been fantastic. "
            "We've been using the SSO integration and the audit log features heavily — "
            "they've been a game changer for our compliance team.\n\n"
            "Quick question: we're planning to add another 50 seats next quarter. "
            "What's the best way to handle that expansion?\n\n"
            "Thanks,\nJordan"
        ),
        "timestamp": "2024-01-15T14:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    # h005: Partner SLA breach. Subject says "API rate limiting" — looks like a tech
    # support ticket. But the body reveals: Business plan SLA violation, partner's own
    # customers affected, P1 designation. Trap: agents classify as technical_support/reply_now.
    # Correct: urgent/technical_support/escalate — SLA breach requires engineering escalation.
    {
        "id": "h005",
        "subject": "Re: API rate limiting affecting our integration",
        "sender": "tech.lead@partner-company.com",
        "body": (
            "We've been hitting your API rate limits consistently for the past week "
            "during our peak hours (2-4pm UTC). We're on the Business plan which should "
            "allow 10k req/min but we're seeing 429s at around 6k req/min. "
            "I've attached our request logs. This is a P1 for us — our product depends "
            "on your API and we have an SLA with our own customers. "
            "If this isn't resolved by EOD we will need to invoke the SLA penalty clause."
        ),
        "timestamp": "2024-01-15T12:00:00Z",
        "has_attachment": True,
        "thread_length": 3,
    },
    # h006: Viral social media crisis. Sender is internal monitoring system, subject is
    # neutral. Trap: agents classify as internal/archive or delegate. Correct: this is
    # an urgent customer_complaint (public-facing) requiring immediate reply_now.
    {
        "id": "h006",
        "subject": "Public tweet about our service going viral",
        "sender": "social.monitoring@company.com",
        "body": (
            "Alert: A tweet from @TechInfluencer (2.1M followers) is going viral. "
            "They posted: 'Just lost 3 hours of work because [YourProduct] deleted my "
            "project without warning. No backup, no notice. Avoid this company.' "
            "Currently at 4,200 retweets and climbing. Several journalists have replied. "
            "We need an immediate public response and to investigate the account."
        ),
        "timestamp": "2024-01-15T17:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    # h007: Automated security alert — dry, technical, easy to dismiss as routine.
    # Trap: agents archive as internal/low because it looks like a standard alert.
    # Hidden signal: 3 failed sudo attempts from Belarus IP on prod-db-01 at 3am
    # = active brute force on production database. Correct: urgent/technical_support/escalate.
    {
        "id": "h007",
        "subject": "Security alert: failed authentication attempts on prod-db-01",
        "sender": "security-alerts@monitoring.internal",
        "body": (
            "Automated security notification\n\n"
            "Host: prod-db-01.internal\n"
            "Event: 3 consecutive failed sudo authentication attempts\n"
            "User: root\n"
            "Source IP: 185.220.101.47 (Tor exit node, geo: Belarus)\n"
            "Time: 03:14:22 UTC\n"
            "Previous successful login: 2024-01-14 09:22 UTC (known IP)\n\n"
            "This is alert #3 of 3 before account lockout threshold.\n"
            "No action has been taken automatically."
        ),
        "timestamp": "2024-01-15T03:15:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    # h008: Aggressive refund demand — angry tone, legal threats, sounds urgent.
    # Trap: agents escalate or reply_now because of the threatening language.
    # Hidden signal: invoice number format invalid, sender is gmail, "contract signed
    # March 2025" is in the future relative to the email date, amount is suspiciously
    # round. This is a social engineering / fraud attempt. Correct: low/spam/delete.
    {
        "id": "h008",
        "subject": "URGENT: Unauthorized charge - demand immediate refund of $5,000",
        "sender": "angry.customer2024@gmail.com",
        "body": (
            "I am writing to demand an IMMEDIATE refund of $5,000 that was charged "
            "to my account without authorization. Invoice REF-000000 dated today "
            "does not correspond to any service I agreed to.\n\n"
            "I have a signed contract from March 2025 that explicitly states no "
            "charges above $500 per month. This charge is a clear breach of contract.\n\n"
            "If I do not receive a full refund within 24 hours I will be contacting "
            "my bank, the FTC, and my attorney. I have already drafted the complaint.\n\n"
            "Do not ignore this email."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
]

# Index by ID for fast lookup
EMAIL_BY_ID: Dict[str, Dict[str, Any]] = {e["id"]: e for e in EMAILS}

# Task definitions: which emails belong to each task
TASK_EMAILS: Dict[str, List[str]] = {
    "easy_triage":   ["e001", "e002", "e003", "e004", "e005"],
    "medium_triage": ["m001", "m002", "m003", "m004", "m005", "m006"],
    "hard_triage":   ["h001", "h002", "h003", "h004", "h005", "h006", "h007", "h008"],
}

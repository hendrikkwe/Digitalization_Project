from enum import Enum


class TCFDDomain(Enum):
    Governance = "Governance"
    Strategy = "Strategy"
    RiskManagement = "RiskManagement"
    MetricsTargets = "MetricsTargets"
    NotDefined = "NotDefined"


# risk, opportunity, neutral
class Ron(Enum):
    Risk = "Risk"
    Opportunity = "Opportunity"
    Neutral = "Neutral"

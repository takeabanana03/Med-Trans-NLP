"""Test-set evaluation, error analysis and the ethics checks. Owner: Deevesh.

Every number in report Sections 5 and 6 should be printed or saved by this module.
"""


def evaluate(name, test, probs, val=None, val_probs=None):
    """Report results for one model on the test set.

    Should print and save to config.OUTPUT_DIR:
    - macro F1 (headline), per-class precision/recall/F1, confusion matrix
    - confidence threshold chosen on val so auto-routed errors stay below a target,
      then auto-routed share and accuracy on test (report 6.1)
    - macro F1 by sex and age group, with group sizes (demographics.extract_demographics, report 6.3)
    - % of predictions that change after demographics.swap_pronouns (report 6.3)
    - a few real misrouted notes for the error analysis
    """
    raise NotImplementedError("Deevesh: evaluate")

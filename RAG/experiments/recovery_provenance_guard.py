"""Temporary provenance guard while full source recovery remains in progress."""
from experiments.closure_family_semantics import query_safe as previous_query
FISCAL_BLOCKS=frozenset('Gráfico '+str(n)for n in[101,102,103,104,108,109,110])
def query_with_recovery_guard(db,*,book,row_number):
    result=previous_query(db,book=book,row_number=row_number)
    if book=='2025'and result.get('block')in FISCAL_BLOCKS and result.get('source_sheet')=='Gráfico 100':
        raise LookupError('Withdrawn: source is temperature, not the attributed fiscal measure; no automatic redirection')
    result['recovery_in_progress']=True
    result['provenance_guard']='source_recovery_2026-11-01'
    return result

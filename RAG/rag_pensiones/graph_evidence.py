"""Bounded literal evidence choices; entailment still requires semantic review."""
import copy
import re

def evidence_options(documents,focus,byte_budget=2000,*,query=''):
    """Offer complete literal sentences fairly across sources, ranked by the request."""
    selected=[p for sid in focus for p in documents if p['source_id']==sid]if focus is not None else documents
    terms=set(re.findall(r'\w{4,}',query.casefold()))
    groups=[]
    for passage in selected:
        candidates=[]
        text=passage['text'];body_start=0;offset=0
        # Page headers are layout, not a sentence prefix. Keep canonical text
        # untouched, and do not offer an opening clause cut at a chunk/page edge.
        if passage.get('start_char',0)==0 and passage.get('page_number')is not None:
            for line in text.splitlines(keepends=True)[:3]:
                offset+=len(line)
                if re.fullmatch(r'\s*(?:[^\W\d_]+\s+)?\d{1,2}\D+\d{4}\s+'+re.escape(str(passage['page_number']))+r'\s*',line):
                    body_start=offset;break
        body=text[body_start:];opening=body.lstrip()[:1]
        clipped=bool((body_start or passage.get('start_char',0))and opening.islower())
        # Keep adjoining acronym components together, e.g. "U. S.".
        sentence_end=r'(?:[!?]|\.(?!\s+[A-ZÁÉÍÓÚÜÑ]{1,3}\.))(?=\s|$)'
        for index,match in enumerate(re.finditer(r'\S[\s\S]*?(?:'+sentence_end+r'|$)',body)):
            if index==0 and clipped:continue
            quote=match.group().strip();words=re.findall(r'\w+',quote)
            size=len(quote.encode('utf-8'))
            if len(quote)<12 or quote[-1]not in '.!?'or not words or size>min(900,byte_budget):continue
            if any(candidate[2]==quote for candidate in candidates):continue
            if sum(any(c.isalpha()for c in w)for w in words)<len(words)/2:continue
            score=len(terms&set(w.casefold()for w in words))
            candidates.append((score,match.start(),quote))
        candidates.sort(key=lambda item:(-item[0],item[1]))
        if candidates:groups.append((passage,candidates))
    groups.sort(key=lambda group:-group[1][0][0])
    result=[];used=0
    while groups:
        remaining=[]
        for passage,candidates in groups:
            while candidates:
                _,_,quote=candidates.pop(0);size=len(quote.encode('utf-8'))
                if used+size>byte_budget:continue
                result.append(dict(evidence_id='e'+str(len(result)+1),source_id=passage['source_id'],quote=quote))
                used+=size
                break
            if candidates:remaining.append((passage,candidates))
        groups=remaining
    return result

def attach_evidence(claims,options,*,rejections=None):
    table={o['evidence_id']:o for o in options};result=[]
    for index,original in enumerate(claims):
        claim=copy.deepcopy(original)
        if any(e['evidence_id']not in table for e in claim['evidence']):
            if rejections is not None:
                rejections.append(dict(need_id=claim['need_id'],claim_index=index,stage='generate',
                    code='unknown_evidence_id',reason='Claim selected evidence not offered by the source selector'))
            continue
        claim['evidence']=[dict(source_id=table[e['evidence_id']]['source_id'],quote=table[e['evidence_id']]['quote'])for e in claim['evidence']]
        result.append(claim)
    return result

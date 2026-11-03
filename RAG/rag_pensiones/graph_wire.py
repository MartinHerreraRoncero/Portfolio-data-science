"""Reversible documentary-review wire views; canonical state stays untouched."""
import copy

NOTE='needs: common se añade a cada rows siguiendo columns. evidence_refs referencia evidence_table (source_id,quote literales).'
FRAGMENT_NOTE='text_fragments concatena strings y quote_ref en orden, sin omisiones; quote_ref usa la quote de evidence_table.'


def encode_review(payload,*,fragments=False):
    result=copy.deepcopy(payload);needs=result['needs'];keys=list(needs[0])if needs else[]
    if any(set(row)!=set(keys)for row in needs):raise ValueError('Nonuniform review need metadata')
    common={key:needs[0][key]for key in keys if all(row[key]==needs[0][key]for row in needs)}
    columns=[key for key in keys if key not in common]
    result['needs']=dict(common=common,columns=columns,rows=[[row[key]for key in columns]for row in needs])
    evidence={}
    for claim in result['claims']:
        refs=[]
        for item in claim.pop('evidence'):
            ref=next((key for key,value in evidence.items()if value==item),None)
            if ref is None:ref='e'+str(len(evidence));evidence[ref]=item
            refs.append(ref)
        claim['evidence_refs']=refs
    result['evidence_table']=evidence
    if fragments:
        for passage in result['passages']:
            text=passage['text'];matches=[]
            for ref,item in evidence.items():
                if item['source_id']!=passage['source_id']:continue
                start=0
                while item['quote']:
                    found=text.find(item['quote'],start)
                    if found<0:break
                    matches.append((found,found+len(item['quote']),ref));start=found+1
            matches.sort()
            if any(left[1]>right[0]for left,right in zip(matches,matches[1:])):continue
            parts=[];cursor=0
            for start,end,ref in matches:
                if start>cursor:parts.append(text[cursor:start])
                parts.append(dict(quote_ref=ref));cursor=end
            if cursor<len(text):parts.append(text[cursor:])
            if matches:passage.pop('text');passage['text_fragments']=parts
    if decode_review_payload(result)!=payload:raise ValueError('Review wire roundtrip differs from canonical payload')
    return result,NOTE+(' '+FRAGMENT_NOTE if fragments else'')


def decode_review_payload(wire):
    result=copy.deepcopy(wire);table=result['needs'];columns=table['columns']
    if len(columns)!=len(set(columns))or set(columns)&set(table['common']):raise ValueError('Invalid review columns')
    if any(len(row)!=len(columns)for row in table['rows']):raise ValueError('Invalid review row')
    result['needs']=[dict(table['common'],**dict(zip(columns,row)))for row in table['rows']]
    evidence=result.pop('evidence_table')
    for claim in result['claims']:claim['evidence']=[copy.deepcopy(evidence[ref])for ref in claim.pop('evidence_refs')]
    for passage in result['passages']:
        if 'text_fragments'in passage:
            parts=[]
            for item in passage.pop('text_fragments'):
                if isinstance(item,str):parts.append(item)
                else:
                    if set(item)!={'quote_ref'}:raise ValueError('Invalid review fragment')
                    quote=evidence[item['quote_ref']]
                    if quote['source_id']!=passage['source_id']:raise ValueError('Cross-source review fragment')
                    parts.append(quote['quote'])
            passage['text']=''.join(parts)
    return result

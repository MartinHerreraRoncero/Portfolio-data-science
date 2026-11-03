"""Unit families for discovery and planning; original SQL units stay literal.

A family describes dimensions, not interchangeable definitions or scales. It
never converts a value, denominator, population or million/thousand amount.
"""

UNIT_FAMILIES=('unspecified','percent_gdp','percentage_points_gdp','percent',
               'percentage_points','euros','persons','pensions_count')


def published_variation(sql_unit,quantity_role=''):
    """Read canonical source metadata, never words from the user's question."""
    label=str(sql_unit).casefold()+' '+str(quantity_role or'').casefold()
    return any(word in label for word in ['percentage_points','growth','change','variation'])


def unit_family(sql_unit,quantity_role=''):
    if not isinstance(sql_unit,str):return None
    unit=sql_unit.casefold()
    if 'signed_plot'in unit or any(t in str(quantity_role or'') for t in ['scatter','paired_coordinates']):return None
    if unit=='percent_gdp':return 'percent_gdp'
    if unit.startswith(('percentage_points_gdp','interannual_percentage_points_gdp')):
        return 'percentage_points_gdp'
    if unit=='percentage_points'or unit.startswith('percentage_points_'):return 'percentage_points'
    if unit=='percent'or unit.startswith('percent_'):return 'percent'
    if 'euro'in unit:return 'euros'
    if 'pension'in unit and 'percent'not in unit:return 'pensions_count'
    if any(t in unit for t in ['person','population','resident']):return 'persons'
    return None


def unit_matches(requested_family,sql_unit,quantity_role='',*,allow_level_operands=False):
    if requested_family not in UNIT_FAMILIES:return False
    if requested_family=='unspecified':return True
    actual=unit_family(sql_unit,quantity_role)
    if actual==requested_family:return True
    return allow_level_operands and (requested_family,actual)in {
        ('percentage_points_gdp','percent_gdp'),('percentage_points','percent')}

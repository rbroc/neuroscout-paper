def _check_nv(m):
    """ Check NV for at least 1 effect and variance at group level in one collection """
    ups = m.get_uploads()
    valid = []
    up = ups[-1]
    for up in ups:
        effects = [f for f in up['files'] if 'effect' in f['basename']]
        variances = [f for f in up['files'] if 'variance' in f['basename']]
        if effects and variances and all([f for f in effects if f['status'] == 'OK']) and all([f for f in variances if f['status'] == 'OK']):
            valid.append(up)
    return any(valid)
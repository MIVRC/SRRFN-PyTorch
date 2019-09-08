def set_template(args):
    # Set the templates here

    if args.template.find('EDSR') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MSRN') >= 0:
        args.model = 'MSRN'
        args.n_blocks = 8
        args.n_feats = 64
        args.chop = True

    if args.template.find('SRRFN') >= 0:
        args.model = 'SRRFN'
        args.n_resgroups = 5
        args.n_resblocks = 10
        args.n_feats = 64
        args.recursive = 4
        args.chop = True

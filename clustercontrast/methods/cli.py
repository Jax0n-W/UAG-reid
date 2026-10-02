"""Shared frozen method flags and experiment naming."""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError("expected a boolean value")


def add_method_arguments(parser):
    parser.add_argument('--use-rahp', action='store_true')
    parser.add_argument('--rahp-beta', type=float, default=0.25)
    parser.add_argument('--rahp-knn', type=int, default=20)
    parser.add_argument('--rahp-alpha', type=float, default=0.5)
    parser.add_argument('--use-cesa', action='store_true')
    parser.add_argument('--cesa-rho', type=float, default=0.8)
    parser.add_argument('--cesa-eta', type=float, default=0.1)
    parser.add_argument('--cesa-lineage-thr', type=float, default=0.5)
    parser.add_argument('--cesa-warmup', type=int, default=5)
    return parser


def experiment_tag(args):
    if args.use_rahp and args.use_cesa:
        return 'rahp_cesa'
    if args.use_rahp:
        return 'rahp'
    if args.use_cesa:
        return 'cesa'
    return 'baseline'


def validate_method_args(args):
    if not 0.0 < args.rahp_beta <= 1.0:
        raise ValueError('--rahp-beta must be in (0, 1]')
    if args.rahp_knn < 1 or not 0.0 <= args.rahp_alpha <= 1.0:
        raise ValueError('--rahp-knn must be positive and --rahp-alpha in [0, 1]')
    if not 0.0 <= args.cesa_rho < 1.0 or args.cesa_eta < 0.0:
        raise ValueError('--cesa-rho must be in [0, 1) and --cesa-eta nonnegative')
    if not 0.0 <= args.cesa_lineage_thr <= 1.0 or args.cesa_warmup < 1:
        raise ValueError('--cesa-lineage-thr must be in [0, 1] and --cesa-warmup positive')
    if (args.use_rahp and not getattr(args, 'stage2_only', False)
            and getattr(args, 'memorybank', None) != 'CMhybrid'):
        raise ValueError('RAHP Stage1 requires --memorybank CMhybrid.')
    batch_size = getattr(args, 'batch_size', None)
    num_instances = getattr(args, 'num_instances', None)
    if batch_size is not None and num_instances is not None and num_instances > 0:
        if batch_size < 2 * num_instances or (batch_size // 2) % num_instances:
            raise ValueError(
                '--batch-size must provide complete identity groups in both domain '
                'loaders: require batch_size >= 2 * num_instances and '
                '(batch_size // 2) % num_instances == 0.')

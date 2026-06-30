from model.agents.BehaviorDDPG import BehaviorDDPG


class HAC(BehaviorDDPG):
    @staticmethod
    def parse_model_args(parser):
        parser = BehaviorDDPG.parse_model_args(parser)
        parser.add_argument('--hyper_actor_coef', type=float, default=0.1,
                            help='hyper-action actor loss coefficient')
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.hyper_actor_coef = args.hyper_actor_coef

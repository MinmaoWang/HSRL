from model.agents.DDPG import DDPG


class PGRA(DDPG):
    @staticmethod
    def parse_model_args(parser):
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--inverse_lr', type=float, default=1e-4,
                            help='inverse model learning rate')
        parser.add_argument('--inverse_decay', type=float, default=1e-4,
                            help='inverse model weight decay')
        parser.add_argument('--inverse_hidden_dims', type=int, nargs='+', default=[128],
                            help='hidden layer sizes for inverse model')
        parser.add_argument('--inverse_dropout_rate', type=float, default=0.1,
                            help='dropout rate of inverse model')
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.inverse_lr = args.inverse_lr
        self.inverse_decay = args.inverse_decay

import torch
from torch import nn

from models_set import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(args, pretrain=True):
    assert opt.model in [
        'resnet', 'preresnet', 'resnext', 'densenet'
    ]

    if args.model == 'resnet':
        assert args.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if args.model_depth == 10:
            model = resnet.resnet10(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 18:
            model = resnet.resnet18(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 34:
            model = resnet.resnet34(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 50:
            model = resnet.resnet50(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 101:
            model = resnet.resnet101(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 152:
            model = resnet.resnet152(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 200:
            model = resnet.resnet200(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)

    elif args.model == 'resnext':
        assert args.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if args.model_depth == 50:
            model = resnext.resnet50(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                cardinality=args.resnext_cardinality,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 101:
            model = resnext.resnet101(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                cardinality=args.resnext_cardinality,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 152:
            model = resnext.resnet152(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                cardinality=args.resnext_cardinality,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)

    elif args.model == 'preresnet':
        assert args.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if args.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif args.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=args.n_classes,
                shortcut_type=args.resnet_shortcut,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)

    elif args.model == 'densenet':
        assert args.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if args.model_depth == 121:
            model = densenet.densenet121(
                num_classes=args.n_classes,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=args.n_classes,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=args.n_classes,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=args.n_classes,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)

    if pretrain:
        print('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path)
        assert args.arch == pretrain['arch']

        model.load_state_dict(pretrain['state_dict'])

        if args.model == 'densenet':
            model.classifier = nn.Linear(
                model.classifier.in_features, args.n_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, args.n_finetune_classes)

        parameters = get_fine_tuning_parameters(model, args.ft_begin_index)
        return model, parameters

    return model, model.parameters()

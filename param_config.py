import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--src_name", type=str, default="mnist32_60_10,mnistm32_60_10,usps32,syn32")
parser.add_argument("--trg_name", type=str, default="svhn")
parser.add_argument("--data_path", type=str, default="../datasets")
parser.add_argument("--data_name", type=str, default="digitFive", help="officeCaltech10 or digitFive")
parser.add_argument("--config", default="svhn.yaml")
parser.add_argument("--data_format", default="mat")
parser.add_argument("--sample_size", type=int, default=5)

parser.add_argument("--log_path", type=str, default="log_model")
parser.add_argument("--model_name", type=str, default="STEM")
parser.add_argument("--process", type=str, default="stem_main")
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--num_iters", type=int, default=40000)
parser.add_argument("--summary_freq", type=int, default=1000)
parser.add_argument("--phase1_iters", type=int, default=0)

parser.add_argument("--src_class_trade_off", type=float, default=1.0)
parser.add_argument("--domain_trade_off", type=float, default=1.0)
parser.add_argument("--trg_trade_off", type=float, default=0.1)
parser.add_argument("--trg_ent_src_domain_trdoff", type=float, default=0.1)
parser.add_argument("--trg_ent_trade_off", type=float, default=0.0)
parser.add_argument("--gen_trade_off", type=float, default=0.0)
parser.add_argument("--mimic_troff", type=float, default=0.1)
parser.add_argument("--src_domain_trade_off", type=str, default="1.0, 1.0, 1.0, 1.0")
parser.add_argument("--loss_trade_off", type=list, default=[1.0, 1.0, 1.0])

parser.add_argument("--inorm", type=bool, default=True)
parser.add_argument("--save_grads", type=bool, default=False)
parser.add_argument("--cast_data", type=bool, default=True)
parser.add_argument("--only_save_final_model", type=bool, default=True)
parser.add_argument("--data_augmentation", type=bool, default=False)
parser.add_argument("--num_classes", type=int, default=10)

parser.add_argument("--using_y_C_model", type=bool, default=False)
parser.add_argument("--adding_node_domain_disc", type=int, default=1)

parser.add_argument("--save_latent", type=bool, default=False)
parser.add_argument("--log_tail", type=str, default=None)
parser.add_argument("--note", type=str, default=None)

args = vars(parser.parse_args())
# print(type(args))